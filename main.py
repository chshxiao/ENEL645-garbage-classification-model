import sys
import os
import torch
from PIL import Image
from matplotlib import pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import GarbageModel, FineTuneType


# check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)


import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR


# read the files
root_path = '/work/TALC/enel645_2024f/garbage_data'

train_folder = '/CVPR_2024_dataset_Train'
val_folder = '/CVPR_2024_dataset_Val'
test_folder = '/CVPR_2024_dataset_Test'

train_path = root_path + train_folder
val_path = root_path + val_folder
test_path = root_path + test_folder


# data transformation
data_transform = transforms.Compose([
  # transforms.RandomResizedCrop([224, 224]),
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip(),
  transforms.ColorJitter(brightness=0.3, contrast=0.3),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])           # empirical numbers for resnet
])
data_transform_test = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load datasets
train_dataset = ImageFolder(root=train_path, transform= data_transform)
val_dataset = ImageFolder(root=val_path, transform= data_transform)
test_dataset = ImageFolder(root=test_path, transform= data_transform_test)


# Define batch size and number of workers (adjust as needed)
batch_size = 58
num_workers = 4


# Create data loaders
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
train_size = len(trainloader) * batch_size
val_size = len(valloader) * batch_size
test_size = len(testloader) * batch_size


# print the distribution of the train dataset
total = []
for i, data in enumerate(trainloader, 0):
  labels = data[1].numpy()
  total = np.append(total, labels)
unique_lable, count = np.unique(total, return_counts=True)
print(dict(zip(unique_lable, count)))


# classes: Black, Green Blur, TTD
class_names = train_dataset.classes
print(class_names)
print("Train set:", train_size)
print("Val set:", val_size)
print("Test set:", test_size)


# train iterator can wraps an iterator around dataset for easy access
train_iterator = iter(trainloader)
train_batch = next(train_iterator)
print(train_batch[0].size())
print(train_batch[1].size())


# set up the model
train_model = GarbageModel(num_classes=4, transfer=True)
train_model.to(device)
train_model.PrintModel()


# set up loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(train_model.parameters(), lr = 0.01)
scheduler = ExponentialLR(optimizer, gamma=0.9)


num_epoch = 15
PATH = './garbage_net.pth'              # Path to save the best model
best_loss = 1e+20

for epoch in range(num_epoch):

  # training loop
  train_loss = 0
  train_correct = 0
  train_model.FineTune(FineTuneType.CLASSIFIER)         # freeze all layers except the fc layer

  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data[0].to(device), data[1].to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = train_model(inputs)                   # learn the dataset
    
    loss = criterion(outputs, labels)       # calculate the loss function
    loss.backward()                         # back calculation based on loss function
    optimizer.step()                        # gradient descent

    # update train loss
    train_loss += loss.item()

    # update the train correct
    _, predicted = torch.max(outputs.data, 1)
    train_correct += (predicted == labels).sum().item()

  print(f'{epoch + 1},  train loss: {train_loss/len(trainloader):.3f},', end = ' ')
  print(f'Accuracy of the network on the training images: {100 * train_correct / train_size} %')
  scheduler.step()


  # valiation loop
  val_loss = 0
  val_correct = 0

  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():

    for i, data in enumerate(valloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data[0].to(device), data[1].to(device)

      # forward
      outputs = train_model(inputs)                   # learn the dataset
      loss = criterion(outputs, labels)       # calculate the loss function

      # update validation loss
      val_loss += loss.item()

      # update validation correct
      _, predicted = torch.max(outputs.data, 1)
      val_correct += (predicted == labels).sum().item()

    print(f'{epoch + 1},  validation loss: {val_loss/len(valloader):.3f},', end = ' ')
    print(f'Accuracy of the network on the validation images: {100 * val_correct / val_size} %')
  

  # save the best model if the loss is smaller than the best loss
  if val_loss < best_loss:
    print("Saving model")
    torch.save(train_model.state_dict(), PATH)
    best_loss = val_loss


print()
print('Finished traning')


# test
# get the model
test_model = GarbageModel(num_classes=4, transfer=False)
test_model.load_state_dict(torch.load(PATH))

# testing loop
test_correct = 0

with torch.no_grad():
  for i, data in enumerate(testloader, 0):
    images, labels = data

    # forward 
    outputs = test_model(images)

    # update the test correct
    _,predicted = torch.max(outputs.data, 1)
    test_correct += (predicted == labels).sum().item()

print(f'Accurately classified images: {test_correct}')
print(f'Accuracy of the network on the test images: {100 * test_correct / test_size} %')
