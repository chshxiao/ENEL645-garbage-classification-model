import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import time
import torch.optim as optim
import numpy as np
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


"""
Use the model to predict the testing dataset
This function returns labels and prediction for test dataset in numpy
"""
def predict(model, test_loader, test_size):

  # testing loop
  test_correct = 0
  labels_test = []
  predict_test = []

  with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
      images, labels = data

      # convert labels to numpy and append to labels_test
      labels_test = np.append(labels_test, labels.numpy())

      # forward
      outputs = model(images)

      # update the test correct
      _,predicted = torch.max(outputs.data, 1)
      test_correct += (predicted == labels).sum().item()

      # convert prediction to numpy and append to predict_test
      predict_test = np.append(predict_test, predicted.numpy())

  print(f'Accurately classified images: {test_correct}')
  print(f'Accuracy of the network on the test images: {100 * test_correct / test_size} %')

  return labels_test, predict_test


"""
Get prediction from the image model or the text model
Results are labels and probabilities of each object in numpy array n * 4
n - number of object - number of images in the dataloader
4 - 4 probabilities of each object
"""
def get_prediction(model, data_loader):
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


class EntireGarbageModel():

  def __init__(self, num_classes, transfer=False):

    self.transfer = transfer                  # transfer flag
    self.num_classes = num_classes
    self.logistic_regression_model = None     # logistic regression model to find the weights between two models

    if self.transfer:
      # logistic regression model takes 8 inputs (4 probabilities from image + 4 probabilities from text)
      # and output 4 probabilities representing classes
      self.logistic_regression_model = LogisticRegression(max_iter=10)


  """
  Get prediction from image model and the text model
  Train the logistic regression model with the output probabilities
  """
  def train(self, image_model, img_train_loader, text_model, text_train_loader):

    # get labels and predictions from both models
    img_labels, img_predict = get_prediction(image_model, img_train_loader)
    text_labels, text_predict = get_prediction(text_model, text_train_loader)


    # Stack the probabilities from both models to create meta-classifier features
    inputs = np.concatenate([img_predict, text_predict], axis=1)
    print(inputs)
    labels = np.concatenate([img_labels, text_labels])


    # Train a simple Logistic Regression as the meta-classifier
    self.logistic_regression_model.fit(inputs, labels)


  def evaluate(self, image_model, img_test_loader, text_model, text_test_loader):

    # get labels and predictions from both models
    img_labels, img_predict = get_prediction(image_model, img_test_loader)
    text_labels, text_predict = get_prediction(text_model, text_test_loader)


    # Stack the probabilities from both models to create meta-classifier features
    inputs = np.concatenate([img_predict, text_predict], axis=1)
    labels = np.concatenate([img_labels, text_labels])


    # predict from logistic regression model
    outputs = self.logistic_regression_model.predict(inputs)


    # calculate the accuracy