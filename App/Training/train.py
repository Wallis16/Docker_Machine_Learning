# Code source: Jaques Grobler
# License: BSD 3 clause
# Code from https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
from numpy import savetxt

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
#diabetes_X = diabetes_X[:, np.newaxis, 2]

diabetes_X = np.expand_dims(diabetes_X[:, 2], axis=1)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

data_test = np.concatenate((diabetes_X_test, np.expand_dims(diabetes_y_test, axis=1)),axis=1)

savetxt('data_test.csv', data_test, delimiter=',')

#print(type(diabetes_X_test))
#np.save("X_test.npy",diabetes_X_test)

#
dump(regr, 'Model/model.joblib') 
#regr_loaded = load('C:/Users/dioge/Desktop/Docker_Machine_Learning/Model/model.joblib') 

# Make predictions using the testing set
#diabetes_y_pred = regr_loaded.predict(diabetes_X_test)

diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

"""Coefficients: [938.23786125]
Mean squared error: 2548.07
Coefficient of determination: 0.47"""