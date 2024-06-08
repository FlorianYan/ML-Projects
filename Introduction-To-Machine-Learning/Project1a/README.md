# Ridge Regression with Cross-Validation
## Overview
In this project, we impletented Ridge Regression with 10-fold Cross-Validation. It is designed to compute and report the Root Mean Squared Error (RMSE) for five different regularization parameters: 0.1, 1, 10, 100, and 200. The objective is to analyze how the regularization strength affects the predictive accuracy of the model.

## Data
The script operates on a dataset provided in `train.csv`, which includes labels and 13 features per datapoint. This dataset is used throughout the cross-validation process to train and test the ridge regression model.

## Implementation
- **fit(X, y, lam)** fits the ridge regression model using the provided data and regularization parameter.
- **calculate_RMSE(w, X, y)** calculates the RMSE between the predicted values and actual labels.
- **average_LR_RMSE(X, y, lambdas, n_folds)** conducts the cross-validation, calculating average RMSE for each lambda value.

The implementation leverages the scikit-learn library for the regression model and cross-validation functionality. The final RMSE results for each regularization strength are saved to `results.csv`.
