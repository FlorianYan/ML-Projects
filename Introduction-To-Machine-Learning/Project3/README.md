# Food Taste Similarity Prediction
## Overview
This project, part of an open-ended task in a machine learning course, aims to predict food taste similarity based on images and human judgments. 
It involves analyzing a dataset of images of 10,000 dishes and a set of triplets indicating human taste preferences. 
The task requires predicting for each triplet (A, B, C) whether dish A is more similar in taste to dish B or C based on their images. 
Output 1, if A is tastes more similar to B, output 0, if A tastes more similar to C.
## Data
- **Images**: 10 000 images of different dishes (not included here).
- **Train Triplets (`train_triplets.txt`)**: Human annotated triplets indicating which two dishes are more similar in taste to each other.
- **Test Triplets (`test_triplets.txt`)**: Triplets for which predictions of taste similarity need to be made.
- `results.txt` contains the predictions for each test triplet.

## Implementation
We first extract the embeddings for the dataset using a pretrained vision model, before training a MLP on top. We make heavy use of the `PyTorch`library.
### Extracting the Embeddings using RESNET50
We transform the input images in order to pass them to RESNET50 and extract the embeddings from the fully connected layer. Here we used RESNET50 with default weights.
### Model selection and training
We tested different Multilayer Perceptron's (MLP's) using Stochastic Gradient Descend (SGD) and BCEWithLogitsLoss for optimization. 
We specifically choose BCEWithLogitsLoss, which combines a signmoid layer with the Binary Cross-Entropy Loss (BCELoss), as it possesses more numerically stable properties than combining them separately.
To avoid exploding or vanishing gradients, we experiemented with LeakyRelu and Batch Normalizations.
