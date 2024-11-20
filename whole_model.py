import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, vgg16
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import time
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

from image_model import *
from text_model import *


class EntireGarbageModel():

  def __init__(self, num_classes, transfer=False):

    self.transfer = transfer                  # transfer flag
    self.num_classes = num_classes
    self.logistic_regression_model = None     # logistic regression model to find the weights between two models

    if self.transfer:
      # logistic regression model takes 8 inputs (4 probabilities from image + 4 probabilities from text)
      # and output 4 probabilities representing classes
      self.logistic_regression_model = LogisticRegression(max_iter=1000)
  

  """
  Get prediction from image model and the text model
  Train the logistic regression model with the output probabilities
  """
  def train(self, image_model, img_train_loader, text_model, text_train_loader, device):

    # get labels and predictions from both models
    img_labels, img_predict = image_model_get_probability(image_model, img_train_loader, device)
    text_labels, text_predict = text_model_get_probability(text_model, text_train_loader, device)


    # Stack the probabilities from both models to create meta-classifier features
    inputs = np.concatenate([img_predict, text_predict], axis=1)
    labels = img_labels
    print(inputs.shape)
    print(labels.shape)


    # Train a simple Logistic Regression as the meta-classifier
    self.logistic_regression_model.fit(inputs, labels)
  

  def evaluate(self, image_model, img_test_loader, text_model, text_test_loader, device):

    # get labels and predictions from both models
    img_labels, img_predict = image_model_get_probability(image_model, img_test_loader, device)
    text_labels, text_predict = text_model_get_probability(text_model, text_test_loader, device)


    # Stack the probabilities from both models to create meta-classifier features
    inputs = np.concatenate([img_predict, text_predict], axis=1)
    labels = img_labels


    # predict from logistic regression model
    outputs = self.logistic_regression_model.predict(inputs)


    # calculate the accuracy
    accuracy = accuracy_score(labels, outputs)
    print("logistic regression classifier accuracy:", accuracy)


def entire_model_predict(regression_model, image_model, img_test_loader, text_model, text_test_loader, device):

  # testing loop
  test_correct = 0
  labels_test = []
  predict_test = []

  # get labels and predictions from both models
  img_labels, img_predict = image_model_get_probability(image_model, img_test_loader, device)
  text_labels, text_predict = text_model_get_probability(text_model, text_test_loader, device)

  with torch.no_grad():
    # Stack the probabilities from both models to create meta-classifier features
    inputs = np.concatenate([img_predict, text_predict], axis=1)
    labels = img_labels

    # predict from logistic regression model
    outputs = regression_model.logistic_regression_model.predict(inputs)

  print(labels)
  print(outputs)
  return labels, outputs



"""
Get prediction from the image model or the text model
Results are labels and probabilities of each object in numpy array n * 4
n - number of object - number of images in the dataloader
4 - 4 probabilities of each object
"""
def get_prediction(model, data_loader, device):
  all_probs = []
  all_labels = []

  with torch.no_grad():
    for i, data in enumerate(data_loader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)

      outputs = model(inputs)               # get the output from the model

      probs = nn.functional.softmax(outputs, dim=1)       # softmax to output the probabilities

      # convert labels to numpy and append to all_labels
      all_labels.append(labels.cpu().numpy())
      # convert probabilities to numpy and append to all_probs
      all_probs.append(probs.cpu().numpy())

  print(np.vstack(all_probs))

  return all_labels, np.vstack(all_probs)
