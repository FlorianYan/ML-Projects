# Generating Scores From User Reviews
## Overview
In this project, given a dataset containing the review's title and its content, we infer the score of the product, which is a continuous value between 0 and 10. 
We use Natural Language Processing (NLP) models for this task.

## Data
`train.csv`contains the training data with title text, content text and score (between 0 and 10).

## Implementation
We pass the input (title and content) using  pretrained transformers in order to extract embeddings. 
We then train a Multilayer Perceptron (MLP) on top, to predict the score.
### Extracting the embeddings using ALBERT
We use Hugging Face libraries to obtain the pretrained transformer ALBERT, a lite version of Google's BERT. 
More sophistcated models such as BERT-large or Open AI's GPT-2 are not needed as we balance tradeoffs between model performance and computational cost.
### Model Selection and Training
We trained a 4 layer MLP using the ADAM optimization algorithm and the MSE loss.
o avoid overfitting, we applied regularization techniques including Dropout and Batch Normalization.
