# Arabic-Handwritten-Character-Recognition
# Deep Lerning Project

# Overview
In this deep learning project we are trying to use supervised method and train several neural network classifiers with the goal of achieving above 95% accuracy. We first tried to train several fully connected Neural Network with different activation functions including Sigmoid, Relu and Tanh in hope of reducing errors and obtaining higher accuracy. Since the accuracies resulted from first three models (two layers fully connected with 500 nodes on each layer) were not satisfying,we then trained a basic Convolutional Neural Network and we could achieve 95.6% accuracy on the test dataset. Afterwards, we tried a full VGG network on the data which resluted in a remarkably low accuracy. this can be becasue of the high number of layers and relatively small size of the prictures, becasue in the researches, the VGG model, that we used,mostly has been used for relative big pictures.Hence, we developed a minimized VGG model which resulted in an acceptable accuracy (0.94). Finally, we tried another VGG model including a dropout layer which resulted in 0.96 accuracy.

# Motivation of the Project
Arabic handwritten characters recognition has been a consistent challenge in the field of computer vision due to the variation in different indivisual handwriting. In addition, the handwriting of each indivisual can be different to some extent each time. Previous iterations of this work has included the use of support vector machines [2] , but now has transitioned to the use of artificial neural networks to challenge this task. Recent studies have shown the promise with these networks, with studies showing the use of a convolutional neural network [1]–[4] . We would like to reproduce this work with the aim of showing the difference in activation function for simple networks in this recognition task, and creating a convolutional neural network specifically for Arabic letter recognition.

# Related work
A project has been done on arabic handwritten letters by Najwa Altwaijry published on June,2020.In that project, there were 591 participants including children aged between 7-12. They trained the model on both Hijja and Arabic Handwritten Character Dataset(AHCD),for which they have had accurasy of 88% and 97% perecent, respectively. their model is including 5 convolutional, 3 maxpooling and 3 fully connected layers with softmax output layer.
https://link.springer.com/article/10.1007/s00521-020-05070-8

# Dataset
The dataset for this project is called “The Arabic Handwritten Characters Dataset”, downloaded from “Kaggle”. The dataset contains 16,800 labeled grayscale images of characters of "32X32",written by 60 participants from different age, all in the form of CSV file. Each character (from “alef” to “yeh”) is written ten times and in two different forms, by each participant. Each training and test dataset have total of 1025 columns. From this 1025 columns, one column contains the class labels (total of 28 classes) and 1024 columns contain pixel-values of each associated character image.

# Required libraries
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython.display import clear_output
from sklearn.utils.class_weight import compute_class_weight
# Required libraries for reading image and processing it
import csv
import scipy
from scipy import ndimage
