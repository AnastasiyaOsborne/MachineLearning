#_________________________________________________________________________________________________________________________
# Author: Anastasiya Osborne
# Name: Linear Regression_AOsborne.py
# Program location:  ~\OneDrive\Documents\Linear Regression_AOsborne.py

# Start: December 25, 2020
# Released: December 27, 2021
# Task: Implement linear regression with multiple variables to predict the prices of houses
# Data: https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/blob/master/USA_Housing.csv 
# Goal: To predict the house price.
#_________________________________________________________________________________________________________________________

# Notes: 
# Please install the following packages with 'pip install'
# If a program doesn't want to bring me to the home path, in a terminal, say cd ~\OneDrive\Documents\
# From Visual Studio Code terminal, I can hold Ctrl+Shift+P and "tclear" anytime to refresh the space and delete all no longer needed text in a terminal.  

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from mlxtend.plotting import scatterplotmatrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from termcolor import colored

print(colored('==============================START OF PROGRAM=========================================================', 'yellow'))

#url = "https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/blob/master/USA_Housing.csv"
#df1 = pd.read_csv(url)

# Home folder: 
# df1 = pd.read_csv('C:\Users\naste\OneDrive\Documents\Sunayu\
# Problems with reading the file straight from the online location. Had to save it to my home folder
# "C:\Users\naste\OneDrive\Documents\Sunayu\Linear-Regression-Model-for-House-Price-Prediction-master"
# and read the "USA_Housing.csv" file from there. 
url = "USA_Housing.csv"
df1 = pd.read_csv(url)

#1000000	{:,}	1,000,000	Number format with comma separator
#print("{:.2f}".format(3.1415926));
#print("{:.2f}".format(df1[Price]));

#compressed_image = np.clip(compressed_image.astype('uint8'), 0, 255)

print(colored('================================EXPLORATORY DATA ANALYSIS (EDA)=========================================================', 'white'))
#EXPLORATORY DATA ANALYSIS (EDA). EDA helps to visually detect the presence of outliers, the distribution of the data, and the relationship between the features. Raschka, Mirjalili. p. 320

print(colored('FIRST 10 ROWS OF THE DATAFRAME', 'red'))
print(df1.head(10))

print(colored('NAMES AND DATA TYPES OF THE COLUMNS', 'red'))
print(df1.dtypes)
# Avg. Area Income                float64
# Avg. Area House Age             float64
# Avg. Area Number of Rooms       float64
# Avg. Area Number of Bedrooms    float64
# Area Population                 float64
# Price                           float64
# Address                          object
# dtype: object

print(colored('INFORMATION ON THE DATAFRAME. EXPLORATION OF COLUMN NAMES, COUNT, AND DATA TYPES', 'red'))
#.info method provides counts of non-null values for each column of the  dataframe object. However, below I check for missing values just to be sure. 
print(df1.info())

print(colored('COLUMN NAMES', 'red'))
print(df1.columns)
# Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
#      dtype='object')

#To check for missing data:
print(colored('CHECK FOR MISSING DATA. IF SUM OF MISSING VALUES IS ZERO, THERE ARE NONE. ', 'red'))

nulls = df1.isnull().sum().to_frame()
for index, row in nulls.iterrows():
    print(index, row[0])
   
# Avg. Area Income 0
# Avg. Area House Age 0
# Avg. Area Number of Rooms 0
# Avg. Area Number of Bedrooms 0
# Area Population 0
# Price 0
# Address 0

print(colored('==================================VISUALIZATION=========================================================', 'white'))

print(colored('SCATTERPLOT OF VARIABLES', 'red'))
cols = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Price']
scatterplotmatrix(df1[cols].values, figsize= (40, 32), names= cols, alpha= 0.5)
plt.title('Scatterplot of 6 Variables in a Housing File')
plt.tight_layout()
plt.savefig('Scatterplot_of_6_housing_variables.png')

# Display the plot. By the way, do we need to show the large plot, or is it better to just save it? 
#plt.show()

print(colored('PRINTING DATA DESCRIPTION', 'red'))
# Generate summary statistics about the numeric data (6 columns out of 7): count, mean (50th percentile), standard deviation, minimum, 25th percentile, 75th percentile, maximum. 
# I want to print the column LABELS on top of the column descriptions. 
# Column Names: 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Price'.
print(df1.describe())

print(colored('================================LINEAR REGRESSION=========================================================', 'white'))

# Linear regression attempts to draw a straight line that will minimize the difference between the real data and the predictions. 
# In other words, it will minimize "residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.""

X = df1[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = df1['Price']
lm = LinearRegression()
lm.fit(X,y)

print(colored('PRINTING INTERCEPT', 'red'))
print(lm.intercept_)
# -2637299.033328586

print(colored('PRINTING COEFFICIENTS', 'red'))
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)
#                               Coefficient
# Avg. Area Income                  21.578049
# Avg. Area House Age           165637.026941
# Avg. Area Number of Rooms     120659.948816
# Avg. Area Number of Bedrooms    1651.139054
# Area Population                   15.200744

print(colored('Predictions from our Linear Regression Model', 'red'))

predictions = lm.predict(X)
plt.scatter(y,predictions) #Data is in line shape, which means our model has done good predictions.
plt.savefig('line_shape.png')
plt.show()
sns.displot((predictions),bins=50) # Data is in bell shape (Normally Distributed), which means our model has done good predictions.
plt.savefig('bell_shape.png')

# Display the plot
#plt.show()


print(colored('========================Regression Evaluation Metrics=========================================================', 'red'))

# The coefficients
print('Coefficients: \n', lm.coef_)
print('Mean Absolute Error:', metrics.mean_absolute_error(y, predictions))
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, predictions))
#print('Mean Squared Error:', metrics.mean_squared_error(y, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, predictions)))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y, predictions))

# Plot outputs

#plt.scatter(X, y, color='black')
#plt.plot(X, y, color='blue', linewidth=3)
#plt.xticks(())
#plt.yticks(())
#plt.show()

#print('Model intercept:', lm.intercept_)
#print('Model slope:',  lm.coef_[0])

print(colored('================================END OF PROGRAM=========================================================', 'yellow'))

