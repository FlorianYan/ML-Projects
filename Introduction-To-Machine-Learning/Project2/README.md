# Power Price Prediction
## Overview
In this project, we aim to predict electricity prices in Switzerland using data from various countries and different seasons.
It tackles common data preprocessing challenges such as handling missing data and dataset noise.

## Data
- `train.csv `contains seasonal data and electricity prices from multiple countries, with Switzerland's prices as the target variable.
- `test.csv` contains seasonal data and electricity prices from multiple countries, without Switzerland.
- `results.csv`constains the predicted electricity prices for Switzerland 

## Implementation
### Data Preprocessing
The initial phase involves loading data and addressing missing values as well as non-numerical data to ensure robust model training. 
Using pandas for data manipulation, missing entries are imputed using sklearn's `IterativeImputer`. Non-numerical data are transfored using sklearn's `OrdinalEncoder`. 
This approach maintains the integrity and distribution of the dataset.
### Model Development and Selection
We used Gaussian Process Regression due to its effectiveness in capturing nonlinear relationships and its flexibility through the use of various kernels such as linear, polynomial, Matern, and RBF. 
This method is particularly suitable for regression tasks where prediction intervals are narrow, like price forecasting.
### Cross-Validation and Hyperparameter Tuning
Model performance is rigorously evaluated using 5-fold cross-validation, ensuring that the model generalizes well across different subsets of data. 
This step involves testing various combinations of kernels and regularization parameters to identify the best model settings that minimize prediction errors.
### Final Model Training
Once the optimal parameters are identified, the final model is trained on the entire dataset. 
This model is then used to generate predictions for the test dataset.

