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
There are two popular options for image classification model: ResNet50 and VGG16. During the experiment, we found that ResNet50 can only achieve 45% accuracy in validation dataset and 40% accuracy in test dataset. However, VGG16 can achieve 65% accuracy in validation dataset and 60% accuracy in test dataset. Since then, we choose VGG16 as the backbone for image classification model.
