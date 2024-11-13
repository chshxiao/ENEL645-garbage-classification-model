import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, vgg16
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time


class FineTuneType:
  """
  An Enum to indicate which layers we want to train.
  """

  "Train just the classifier layer in case it's different from the newly added layers (feature-extraction)."
  CLASSIFIER = 1
  "Train all the layers (fine-tuning)."
  ALL = 2


class EarlyStopping():
  def __init__(self, threshold=0.15, patience=4):
    self.threshold = threshold          # threshold to determine whether the model is learning
    self.patience = patience            # maximum epochs before early stopping
    self.counter = 0                # count down
    self.early_stop_flag = False
    self.best_loss = 1e+20


  def __call__(self, val_loss):
    # validation loss doesn't decrease
    if val_loss > self.best_loss - self.threshold:
      self.counter += 1

      if self.counter >= self.patience:
        self.early_stop_flag = True


class GarbageModel(nn.Module):

  def __init__(self, num_classes, transfer=False):
    super().__init__()
    self.transfer = transfer                  # transfer flag
    self.num_classes = num_classes
    self.dummy = nn.Parameter(torch.empty(0))           # contain device
    self.pretrained_model = None
    self.classifier = None

    if self.transfer:
      self.pretrained_model = vgg16(weights=True)
      self.classifier = nn.Linear(1000, 4)               # vgg - add a new layer
    else:
      self.pretrained_model = vgg16(weights=None)
      self.classifier = nn.Linear(1000, 4)               # vgg - add a new layer


  def forward(self, x):
    x = self.pretrained_model(x)
    x = self.classifier(x)
    return x


  def FineTune(self, what:FineTuneType):
    # only update the fc layer
    if what is FineTuneType.CLASSIFIER:
      # freeze all parameters
      for param in self.pretrained_model.parameters():
        param.requires_grad = False
      # unfreeze fc layer parameters
      for param in self.classifier.parameters():
        param.requires_grad = True
    # update the whole structure (finetune)
    else:
      for param in self.pretrained_model.parameters():
        param.requires_grad = True

  """
  Train the model in one epoch
  Used on train_multi_epochs_and_save_best_model for iterative training
  Returns train loss and train validation in the format
  [train_loss, train_val]
  """
  def train_one_epoch(self, train_loader, optimizer, train_size, criterion, epoch):
    start_time = time.time()              # start time of this train
    train_loss = 0
    train_correct = 0
    device = self.dummy.device

    for i, data in enumerate(train_loader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data[0].to(device), data[1].to(device)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = self(inputs)

      loss = criterion(outputs, labels)       # calculate the loss function
      loss.backward()                         # back calculation based on loss function
      optimizer.step()                        # gradient descent

      # update train loss
      train_loss += loss.item()

      # update the train correct
      _, predicted = torch.max(outputs.data, 1)
      train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_correct = 100 * train_correct / train_size
    end_time = time.time()                    # end time of the train

    print(f'{epoch + 1}, ellapsed time: {end_time - start_time}')
    print(f'train loss: {train_loss:.3f},', end = ' ')
    print(f'Accuracy of the network on the training images: {train_correct} %')

    return train_loss, train_correct


  """
  Evaluate the model in one epoch
  Used on train_multi_epochs_and_save_best_model for iterative training
  Returns validation loss and validation validation in the format
  [validation_loss, validation_val]
  """
  def validate_one_epoch(self, val_loader, val_size, criterion, epoch):
    start_time = time.time()              # start time of this validation
    val_loss = 0
    val_correct = 0
    device = self.dummy.device

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
      for i, data in enumerate(val_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # forward
        outputs = self(inputs)
        loss = criterion(outputs, labels)       # calculate the loss function

        # update train loss
        val_loss += loss.item()

        # update the train correct
        _, predicted = torch.max(outputs.data, 1)
        val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_correct = 100 * val_correct / val_size
    end_time = time.time()                      # end time of the validation process

    print(f'{epoch + 1}, time ellapsed: {end_time - start_time}')
    print(f'Validation loss: {val_loss:.3f},', end = ' ')
    print(f'Accuracy of the network on the validation images: {val_correct} %')

    return val_loss, val_correct


  """
  Train the model in multiple epochs and save the best model
  Specify parameters for both classifier layer tuning and fine tuning.
  This function returns loss and accuracy in the format:
  [train loss, train accuracy, validation loss, validation accuracy]
  """
  def train_multi_epochs_and_save_best_model(self, train_loader, val_loader, train_size, val_size, criterion, path,
    optimizer, scheduler, epochs,
    finetune_optimizer, finetune_scheduler, finetune_epochs):

    total_train_loss = []
    total_train_correct = []
    total_val_loss = []
    total_val_correct = []

    best_loss = 1e+20
    best_acc = 0
    early_stopping = EarlyStopping()

    # learn the classifier layer
    self.FineTune(FineTuneType.CLASSIFIER)          # only unfreeze the classifier layer

    print("Start training")

    for epoch in range(epochs):
      train_loss, train_correct = self.train_one_epoch(train_loader, optimizer, train_size, criterion, epoch)
      scheduler.step()                  # schedule learning rate
      val_loss, val_correct = self.validate_one_epoch(val_loader, val_size, criterion, epoch)

      # store the results
      total_train_loss.append(train_loss)
      total_train_correct.append(train_correct)
      total_val_loss.append(val_loss)
      total_val_correct.append(val_correct)

      # save the model if the validation accuracy is the highest
      if val_correct > best_acc:
          print("Saving model")
          torch.save(self.state_dict(), path)
          best_acc = val_correct

      # update early stopping class to check if we should skip the loop
      early_stopping(val_loss)
      if early_stopping.early_stop_flag:
        print("Early stopping at epoch: ", epoch)
        break

    # fine tune
    if not early_stopping.early_stop_flag:
      self.FineTune(FineTuneType.ALL)                           # unfreeze all layers

      for epoch in range(finetune_epochs):
        train_loss, train_correct = self.train_one_epoch(train_loader, finetune_optimizer, train_size, criterion, epoch)
        finetune_scheduler.step()
        val_loss, val_correct = self.validate_one_epoch(val_loader, val_size, criterion, epoch)

        # store the result
        total_train_loss.append(train_loss)
        total_train_correct.append(train_correct)
        total_val_loss.append(val_loss)
        total_val_correct.append(val_correct)

        # save the model if the validation accuracy is the highest
        if val_correct > best_acc:
          print("Saving model")
          torch.save(self.state_dict(), path)
          best_acc = val_correct

        # update early stopping class to check if we should skip the loop
        early_stopping(val_loss)
        if early_stopping.early_stop_flag:
          print("Early stopping at epoch: ", epoch)
          break

    # return results
    return total_train_loss, total_train_correct, total_val_loss, total_val_correct