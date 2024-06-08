# Linear Regression with feature transformation
## Overview
This project applies feature engineering techniques and regularization techniques to enhance the performance of linear regression models. By transforming input features into a higher-dimensional space that includes linear, quadratic, exponential, cosine, and constant transformations, the model can capture more complex patterns in the data.

## Data
The train.csv dataset comprises an ID, the target variable `y`, and five predictor variables `x1` to `x5`. The transformation expands these five predictors into 21 features to be used in the regression model, improving the capacity of the model to learn from complex relationships.

## Implementation
- **transform_data(X)** applies feature transformation to the input data, which is given as a matrix.
- **fit(X, y)** computes the weights of the fitted linear regression.

We leverated the closed form solutions of Ridge Regression and LSE solving to avoid numerical instabilities.
