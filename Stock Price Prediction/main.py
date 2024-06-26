# Modules
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Data Preparation
def prepare_data(df, forecast_col, forecast_out, test_size):
    '''This function is for preparing Data'''
    df['label'] = df[forecast_col].shift(-forecast_out)  # creating new column called label with the last forecast_out rows as NaN
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    X_lately = X[-forecast_out:]  # X that will contain the last forecast_out values for future prediction
    X = X[:-forecast_out]  # X that will be used for training and testing

    df.dropna(inplace=True)  # dropping NaN values
    y = np.array(df['label'])  # assigning Y

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response

# Reading the Data

df = pd.read_csv('D:/VS C/ML/Projects/Stock Price Prediction/INR=X.csv')

forecast_col = 'Close'
forecast_out = 5
test_size = 0.2

# Applying Machine Learning for Stock Price Prediction

X_train, X_test, Y_train, Y_test, X_lately = prepare_data(df, forecast_col, forecast_out, test_size) #calling the method were the cross validation and data preperation is in
learner = LinearRegression() # initializing linear regression model

learner.fit(X_train, Y_train) # training the linear regression model

score = learner.score(X_test, Y_test) #testing the linear regression model
forecast = learner.predict(X_lately) #set that will contain the forecasted data
response = {} #creting json object
response['test_score'] = score
response['forecast_set'] = forecast

print(response)