#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

#importing files
df = pd.read_csv('./dataset/housing.csv')

####### preprocessing ################
#checking null values
#print(df.isnull().sum())
#cleaning null values
totalbedrooms = df['total_bedrooms']
df["total_bedrooms"].fillna(0, inplace = True)
#checking null values in all columns
#print(df.isnull().sum())
#converting labels to numberics
#print(df['ocean_proximity'].unique())
dataframe = pd.get_dummies(df['ocean_proximity'])
del df['ocean_proximity']
completeSet = pd.concat([df, dataframe], axis=1)
del completeSet['NEAR OCEAN']
##### splitting training and test set ########

X = completeSet.drop(["median_house_value"], axis=1)
y = completeSet["median_house_value"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(np.sqrt(mse))

print(regressor.score(X_train, y_train))

print(regressor.score(X_test, y_test))

print(regressor.predict(X_test.head(1)))













