# ENEL645-garbage-classification-model

## Introduction

This machine learning model aims to determine the kind of the garbage. Garbages can be divided into four groups: **Green**, **Black**, **Blue**, and **TTR**.

User will provide a picture of the garbage and a short natural language description on it to the model to predict the result.

## Model

This machine learning model consists of three sub models.
1. The first one is an image classification model. The backbone of this model is VGG16. A new fully connected layer is appended to the top of the model.
2. The second model is a natural language model. The backbone is DistilBERT. A linear layer is appended to the top of the model to fit in four categories in our scenario
3. The last model is a logistic regression model. This model is applied upon the image classification model and the natural language model. It takes the prediction from both models and applied them together to find out the weights of each output probability with the best loss result.

## Data
The data for the training consists of 15488 images altogether. The distribution of the images is as follow:

*Black: 2111
*Blue: 4355
*Green: 1991
*TTR: 1743

As can be seen, the dataset is a little bit imbalanced. Since then, **weighted cross entropy loss** replaces **cross entropy loss** to fix the imbalance

## Methodology

### Data Augmentation
To improve the generality of the dataset, we implemented multiple methods including resize, centercrop, affine transformation, colorjitter, and so on. After that, normalization is applied to the tensor

### Model Selection
There are two popular options for image classification model: ResNet50 and VGG16. During the experiment, we found that ResNet50 can only achieve 45% accuracy in validation dataset and 40% accuracy in test dataset. However, VGG16 can achieve 65% accuracy in validation dataset and 60% accuracy in test dataset. Since then, we choose VGG16 as the backbone for image classification model. In terms of the dense layer to the output which is the number of classes, we chose to have one Linear layer
```
self.classifer = nn.Linear(1000, num_classes)
```
Where 1000 is the number of output neurons from the last layer in the VGG16 backbone.

For the text model, we used DistilBERT as our backbone and add a dense layer to output four classes at the end.


## Experiment design

### Data Augmentation
We used the example image augmentation code from ENEL645-F2024 github repository as the beginning. We did several changes on the data augmentation:
1. We found images are taken at different background. Some of them re taken on the white floor while the others are taken on the dark floor. At the same time, the light is also varying among different pictures. Since then, we added a color jitter to generalize the images.
```
transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2),
```
2. Images are taken in different angle and different perspectives. This introduces changes on shape of the objects. Since then, we added RandomAffine and RandomRotation to solve it.
```
transforms.RandomRotation(50),
transforms.RandomAffine(40, scale=(1, 1.2)),
```

### Imbalanced dataset
The distribution of the dataset is not equal. There are more garbages labelled as blue in the dataset. Since then, we applied a weighted cross entropy loss in the training process rather than the normal cross entropy loss. The weight is the ratio of the class in the training dataset. The code is shown below:
```
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
criterion = nn.CrossEntropyLoss(weight=class_weights)
```


### Image Model Selection
We used ResNet50 initially. However, the validation accuracy improves from 30% at the first epoch to 45% after several epochs of training. The validation loss shows the same trendline as it decreases and stops at around 1.5 after several epochs of training. We believe this is due to the reason that ResNet50 doesn't get the features from the images. So we replaced it with VGG16 and the accuracy improved to 65% and the loss decreased to 0.8.

The transfer learning process of image model consists of two parts. In the first part, all the weights are frozen except the dense layer created. The learning rate is set to be 0.001. This process is to fastly optimized the weights in the last layer as all the weights in the backbone are pretrained.

The second part is the fine tune. This is done by unfreezing all parameters. However, the learning rate is set to 1e-6. This is done by setting a finetune flag in the image model class.
``` 
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
```
