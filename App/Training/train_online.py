import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

#url='https://drive.google.com/file/d/1M3eb9oUQnvf4Nsty8Xqg7Y5e25pl-CrR/view?usp=sharing'

def training(url):

    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, header=None)

    #X = df.iloc[:, 0:-1].values
    #y = df.iloc[:,-1:].values

    # Load the diabetes dataset
    diabetes_X, diabetes_y = df.iloc[:, 0:-1].values, np.squeeze(df.iloc[:,-1:].values)

    #print(diabetes_X.shape, diabetes_y.shape)

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

    #
    dump(regr, 'Model/model.joblib')