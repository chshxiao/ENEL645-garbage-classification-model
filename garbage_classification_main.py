from torch.optim.lr_scheduler import ExponentialLR
from whole_model import *
from image_model import *
from text_model import *


# check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)


# read the files
root_path = '/work/TALC/enel645_2024f/garbage_data'

train_folder = '/CVPR_2024_dataset_Train'
val_folder = '/CVPR_2024_dataset_Val'
test_folder = '/CVPR_2024_dataset_Test'

train_path = root_path + train_folder
val_path = root_path + val_folder
test_path = root_path + test_folder


# image model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data transformation
data_transform = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip(),
  transforms.RandomRotation(90),
  transforms.RandomAffine(60, scale=(1, 1.3)),
  transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])           # empirical numbers for resnet
])
data_transform_test = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.Resize(270),                                 # change input size
  transforms.CenterCrop(256),                             # change input size
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load datasets
train_dataset = ImageFolder(root=train_path, transform= data_transform)
val_dataset = ImageFolder(root=val_path, transform= data_transform)
test_dataset = ImageFolder(root=test_path, transform= data_transform_test)


# Define batch size and number of workers (adjust as needed)
batch_size = 32
num_workers = 4


# Create data loaders
img_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
img_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
img_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
train_size = len(img_train_loader) * batch_size
val_size = len(img_val_loader) * batch_size
test_size = len(img_test_loader) * batch_size


# print the distribution of the train dataset
total = []
for i, data in enumerate(img_train_loader, 0):
  labels = data[1].numpy()
  total = np.append(total, labels)
unique_lable, count = np.unique(total, return_counts=True)
print(dict(zip(unique_lable, count)))


# calculate weight for each class in cross entropy loss
class_weights = 1 / count
class_weights_sum = np.sum(class_weights)
class_weights /= class_weights_sum                          # normalize the class weights
class_weights = torch.tensor(class_weights, dtype=torch.float32)           # convert to tensor to initialize loss function
class_weights = class_weights.to(device)


# classes: Black, Green Blur, TTD
class_names = train_dataset.classes
print(class_names)
print("Train set:", train_size)
print("Val set:", val_size)
print("Test set:", test_size)


# train iterator can wraps an iterator around dataset for easy access
train_iterator = iter(img_train_loader)
train_batch = next(train_iterator)
print(train_batch[0].size())
print(train_batch[1].size())


# set up the model
image_model = GarbageModel(num_classes=4, transfer=True)
image_model.to(device)
print(image_model)


# set up loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(image_model.parameters(), lr = 0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9)
finetune_optimizer = torch.optim.AdamW(image_model.parameters(), lr = 1e-6)
finetune_scheduler = ExponentialLR(optimizer, gamma=0.9)


num_epoch = 15
finetune_num_epoch = 5


img_path = './garbage_image_model.pth'              # Path to save the best model
best_loss = 1e+20


# train process
train_loss, train_acc, val_loss, val_acc = image_model.train_multi_epochs_and_save_best_model(
    img_train_loader, img_val_loader, train_size, val_size, criterion, img_path,
    optimizer, scheduler, num_epoch,
    finetune_optimizer, finetune_scheduler, finetune_num_epoch
  )

print()
print('Finished traning')


# test process
# get the model
image_model_test = GarbageModel(num_classes=4, transfer=False)
image_model_test.load_state_dict(torch.load(img_path))
image_model_test.to(device)

# testing loop
labels_test, predict_test = predict(image_model_test, img_test_loader, test_size)

# confusion matrix
print("confusion matrix:")
cm = confusion_matrix(labels_test, predict_test)
print(cm)


# text model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# read train, validation, and test datasets
text_train,labels_train = read_text_files_with_labels(train_path)
text_val,labels_val = read_text_files_with_labels(val_path)
text_test,labels_test = read_text_files_with_labels(test_path)


# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize data
max_len = 24
dataset_train = CustomDataset(text_train, labels_train, tokenizer, max_len)
dataset_val = CustomDataset(text_val, labels_val, tokenizer, max_len)
dataset_test = CustomDataset(text_test, labels_test, tokenizer, max_len)

# Data loaders
batch_size = 16
text_train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
text_val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
text_test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

best_loss = 1e+10 # best loss tracker
best_acc = 0
num_epoch = 1

text_path = './garbage_text_model.pth'

text_model = DistilBERTClassifier(num_classes=4)
text_model.to(device)
print(text_model)


# Training parameters
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()


# train process
train_loss, train_acc, val_loss, val_acc = text_model.train_multi_epochs_and_save_best_model(
    text_train_loader, text_val_loader, train_size, val_size, criterion, text_path,
    optimizer, num_epoch
  )


print()
print('Finished traning')


# Test Evaluation
text_path = './best_text_model.pth'
text_model_test = DistilBERTClassifier(num_classes=4)
text_model_test.load_state_dict(torch.load(text_path))
text_model_test.to(device)

labels_test, predict_test = predict(model, text_test_loader, test_size)
print(f"Accuracy:  {(predict_test == labels_test).sum()/labels_test.size:.4f}")
cm = confusion_matrix(labels_test, predict_test)
print(cm)


# combined model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
model = EntireGarbageModel(num_classes=4, transfer=True)
model.train(image_model_test, img_train_loader, text_model_test, text_train_loader, device)
model.evaluate(image_model_test, img_test_loader, text_model_test, text_test_loader, device)