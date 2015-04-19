__author__ = 'Satish'
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import cross_validation, linear_model
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn import metrics

def readData():
    # Load the data, which is included in sklearn.
    bike_sharing_demand = pd.read_csv('./data/input/train.csv')
    prediction_data = pd.read_csv('data/input/test.csv')
    train_data, train_labels = bike_sharing_demand.ix[:, 'datetime':'windspeed'], bike_sharing_demand.ix[:, 'casual':]
    prediction_data = prediction_data.ix[:, 'datetime':'windspeed']
    np.random.seed(0)
    shuffle = np.random.permutation(np.arange(train_data.shape[0]))
    mini_bike_sharing = bike_sharing_demand.ix[shuffle[:100], :]
    mini_train_data, mini_train_labels = mini_bike_sharing.ix[:, 'datetime':'windspeed'], mini_bike_sharing.ix[:, 'casual':]
    # Let's extract the information
    for dataset in (train_data, prediction_data):
        dataset['hour'] = dataset['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).hour)
        dataset['weekday'] = dataset['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).weekday())
        dataset['month'] = dataset['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).month)
        dataset['year'] = dataset['datetime'].map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).year)

    rs = cross_validation.ShuffleSplit(train_data.shape[0], n_iter=10, random_state=0)
    return rs,train_data,train_labels


# Define the Root-Mean-Squared-Log Error function for scoring predictions
def rmsle(actual_values, predicted_values):
    squared_log_errors = (np.log(np.array(predicted_values) + 1) - np.log(np.array(actual_values) + 1)) ** 2
    mean_squared_errors = np.nansum(squared_log_errors) / len(squared_log_errors)
    return np.sqrt(mean_squared_errors)


