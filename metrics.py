import matplotlib.pyplot as plt
import pandas as pd


"""
Plot loss function vs. epoch
"""
def plot_loss_function(train_loss, val_loss):

  # check if two loss variables have the same dimension
  if len(train_loss) != len(val_loss):
    print("train loss and validation loss have different sizes")
    return

  fig = plt.figure()
  plt.plot(train_loss)
  plt.plot(val_loss)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['Train Loss', 'Validation Loss'])
  plt.title('Loss Function vs. Epoch')


"""
Plot accuracy vs. epoch
"""
def plot_accuracy_function(train_acc, val_acc):

  # check if two accuracy variables have the same dimension
  if len(train_acc) != len(val_acc):
    print("train accuracy and validation accuracy have different sizes")
    return

  fig = plt.figure()
  plt.plot(train_acc)
  plt.plot(val_acc)
  plt.xlabel('epoch')
  plt.ylabel('Accuracy (%)')
  plt.legend(['Train Accuracy', 'Validation Accuray'])
  plt.title('Accuracy vs. Epoch')