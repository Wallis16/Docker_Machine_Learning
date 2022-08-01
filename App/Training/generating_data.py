import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
from numpy import savetxt

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

data = np.concatenate((diabetes_X, np.expand_dims(diabetes_y, axis=1)),axis=1)

# save to csv file
savetxt('data.csv', data, delimiter=',')

#print(diabetes_X.shape, diabetes_y.shape, a.shape)
#print(diabetes_X[0], diabetes_y[0], a[0])