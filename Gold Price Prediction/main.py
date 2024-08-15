"""Importin Modules"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

"""Data collection and Processing"""
# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv("/gld_price_data.csv")

# print first 5 rows in the dataframe
head = gold_data.head()
print(f'\nHead of Dataset: {head}')

# print last 5 rows of the dataframe
tail = gold_data.tail()
print(f"\nTail of Dataset: {tail}")

# number of rows and columns
shape = gold_data.shape
print(f"\nShape of Dataset: {shape}")

# getting some basic informations about the data
info = gold_data.info()
print(f"\nInformation about our data: {info}")

# checking the number of missing values
sum_of_nullV = gold_data.isnull().sum()
print(f"\nSum of null values: {sum_of_nullV}")

# getting the statistical measures of the data
des = gold_data.describe()
print(f"\nStatistical measures: {des}")

"""Correlation"""
correlation = gold_data.corr(numeric_only=True)

# constructing a heatmap to understand the correlatiom
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, 
            square=True, fmt='.1f', \
            annot=True, annot_kws={'size':8}, cmap='Blues')

# correlation values of GLD
print(correlation['GLD'])

# checking the distribution of the GLD Price
sns.displot(gold_data['GLD'], color='green')

"""Splitting th Features and Target"""
X = gold_data.drop(['Date', "GLD"], axis=1)
Y = gold_data['GLD']

# print(Y)
# print(X)

"""Splitting into Training data and Test data"""
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.2, 
                                                    random_state=2)

"""Model Training: Random Forest Regressor"""
regressor = RandomForestRegressor(n_estimators=100)

# training the model
regressor.fit(X_train, Y_train)

"""Model Evaluation"""
# prediction on Test Data
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print(f"R squared error: {error_score}")

"""Compare the Actual Values and Predicted Values in a Plot"""
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()