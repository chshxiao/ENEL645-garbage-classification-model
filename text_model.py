import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Extract text from file names as well as labels
def read_text_files_with_labels(path):
    texts = []                        # a list to store the description of the objects
    labels = []                       # a list to store the labels
    class_folders = sorted(os.listdir(path))  # Assuming class folders are sorted
    print(class_folders)
    # a dict that assign unique labels to each class folder
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}
    print(label_map)
    
    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    file_name_no_ext, _ = os.path.splitext(file_name)     # get rid of the extension .jpg/png
                    text = file_name_no_ext.replace('_', ' ')             # replace all underscore with space
                    text_without_digits = re.sub(r'\d+', '', text)        # remove digits
                    texts.append(text_without_digits)
                    labels.append(label_map[class_name])

    return np.array(texts), np.array(labels)


# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer      # tokenize texts into token IDs
        self.max_len = max_len          # max length for the tokenized length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenization process
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,      # set max length
            return_token_type_ids=False,
            padding='max_length',         # pad text to max length
            truncation=True,              # truncates texts longer than max length
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Define the model
class DistilBERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.drop(pooled_output[:,0])
        return self.out(output)


class EarlyStopping():
  def __init__(self, threshold=0.15, patience=4):
    self.threshold = threshold
    self.patience = patience
    self.counter = 0
    self.early_stop_flag = False
    self.best_loss = 1e+20
  

  def __call__(self, val_loss):
    if val_loss > self.best_loss - self.threshold:
      self.counter += 1

      if self.counter >= self.patience:
        self.early_stop_flag = True


"""
Define training function
Return train loss and correct in the format
[train loss, train correct(%)]
"""
def train(model, iterator, optimizer, criterion, device, train_size):
    model.train()

    total_loss = 0
    total_correct = 0

    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # update the train correct
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == labels).sum().item()

    total_loss = total_loss / len(iterator)
    total_correct = 100 * total_correct / train_size

    return total_loss, total_correct


"""
Define evaluation function
Return validation loss and correct inthe format
[validation loss, validation correct(%)]
"""
def evaluate(model, iterator, criterion, device, val_size):
    model.eval()

    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)

            total_loss += loss.item()

            # update the train correct
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == labels).sum().item()

        total_loss = total_loss / len(iterator)
        total_correct = 100 * total_correct / val_size

    return total_loss, total_correct


def text_model_predict(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # Disable gradient tracking
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)  # Assuming input_ids are in the batch
            attention_mask = batch['attention_mask'].to(device)  # Assuming attention_mask is in the batch

            # Forward pass
            outputs = model(input_ids, attention_mask)

            # Get predictions
            _, preds = torch.max(outputs, dim=1)

            # Convert predictions to CPU and append to the list
            predictions.extend(preds.cpu().numpy())

    return predictions


def text_model_get_probability(model, data_loader, device):

    all_probs = []
    all_labels = []

    with torch.no_grad():  # Disable gradient tracking
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)  # Assuming attention_mask is in the batch

            # Forward pass
            outputs = model(input_ids, attention_mask)

            probs = nn.functional.softmax(outputs, dim=1)       # softmax to output the probabilities

            # convert labels to numpy and append to all_labels
            all_labels = np.concatenate([all_labels, data['label'].cpu().numpy()])
            # convert probabilities to numpy and append to all_probs
            all_probs.append(probs.cpu().numpy())
    
    return all_labels, np.vstack(all_probs)