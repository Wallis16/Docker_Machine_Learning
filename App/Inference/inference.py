import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

#url='https://drive.google.com/file/d/1M3eb9oUQnvf4Nsty8Xqg7Y5e25pl-CrR/view?usp=sharing'
#url test = "https://drive.google.com/file/d/1cORpTUDA_teADieoRSEWKPPQz34nWWPK/view?usp=sharing"

def prediction(url):

    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df = pd.read_csv(url, header=None)

    # Load the diabetes dataset
    diabetes_X_test, diabetes_y_test = df.iloc[:, 0:-1].values, np.squeeze(df.iloc[:,-1:].values)

    regr_loaded = load('Model/model.joblib')

    # Make predictions using the testing set
    diabetes_y_pred = regr_loaded.predict(diabetes_X_test)

    np.save("Results/predicted_values.npy",diabetes_y_pred)