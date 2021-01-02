# Linear Regression
Model is Linear Regression. The goal is to predict the price, given multiple dependent variables.

## Code
1. Plan. Data source was "https://github.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/blob/master/USA_Housing.csv."
 
Raw data was saved in my local folder “C:\Users\naste\OneDrive\Documents\Sunayu”. The file had mistakes when read straight from GitHub. I had to save it in a local folder. 
Model is Linear Regression. The goal is to predict the price, given multiple dependent varia-bles.


2. Gather data. START OF PROGRAM
 Housing data in CSV format were read into Python. 
"C:\Users\naste\OneDrive\Documents\Sunayu\Linear-Regression-Model-for-House-Price-Prediction-master" is the new location for "USA_Housing.csv"data file. 
 
2.1. Exploratory data analysis (EDA).

Housing data in CSV format were read into Python. Linear-Regression-Model-for-House-Price-Prediction-master" is the new location for "USA_Housing.csv"data file. 
The next step was Exploratory Data Analysis (EDA). The purpose of EDA was to visualize the possible presence of outliers, the data distribution, and relationship between the variables.
Steps: - FIRST 10 ROWS OF THE DATAFRAME. Names of the columns: 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'].
Number of rows: 5,000. 

-	NAMES AND DATA TYPES OF THE COLUMNS
-	INFORMATION ON THE DATAFRAME. EXPLORATION OF COLUMN NAMES, COUNT, AND DATA TYPES using .info method that provides counts of non-null values for each column of the dataframe object. However, below I check for missing values just to be sure. 
-	COLUMN NAMES
-	CHECK FOR MISSING DATA. IF SUM OF MISSING VALUES IS ZERO, THERE ARE NONE.

2.2. Visualization.

-	SCATTERPLOT OF VARIABLES
-	PRINTING DATA DESCRIPTION. This step generates summary statistics about the numeric data (6 columns out of 7): count, mean (50th percentile), standard deviation, minimum, 25th percentile, 75th percentile, maximum.

3.  Do training of a machine learning model and debugging.  

LINEAR REGRESSION

X =  'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population'.
y = 'Price'.

-	PRINTING INTERCEPT. 
-	PRINTING COEFFICIENTS
-	Predictions from the Linear Regression Model
-	Regression Evaluation Metrics (mean_absolute_error, mean_squared_error).


The linear regression based on the 5,000 sample of housing data looks like this: 

Housing Price = -2637299.033328586 + 21.578049* Avg. Area Income + 165637.026941* Avg. Area House Age + 120659.948816* Avg. Area Number of Rooms 
+  1651.139054* Avg. Area Number of Bedrooms  + 15.200744* Area Population.                   

## Images

bell_shape.png
line_shape.png

## Prereqisites: 
Raw data were taken from GitHub and saved in my local folder. The file had mistakes when read straight from GitHub. I had to save it in a local folder. 

## The scope of the project. 
