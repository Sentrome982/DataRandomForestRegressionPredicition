import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

### DISPLAY MAXIMUM COLUMNS
pd.options.display.max_columns = None

### OPENING THE CSV FILE
df = pd.read_csv("Datasets/petrol_consumption.csv")

### DISPLAYING THE DATASET
# print(df.head())

### X IS THE FEATURE SET AND y IS THE LABEL SET
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

### SPLITS THE DATA INTO TRAINING AND TESTING SETS (X is data, y is solutions)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### CREATING A STANDARD SCALER OBJECT TO USE FEATURE SCALING ON THE DATA
sc = StandardScaler()

###  AUTOMATICALLY NORMALIZE THE DATA IN THE TRAINING AND TESTING SETS
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### CREATING A RANDOM FOREST REGRESSOR (n=# of trees(train), r_s at 0=algorithm deterministic)
regressor = RandomForestRegressor(n_estimators=200, random_state=0)

### TAKES IN TRAINING SET AND BUILDS A FOREST
regressor.fit(X_train, y_train)

### MAKES PREDICTIONS BASED ON TRAINING
y_pred = regressor.predict(X_test)

### AVERAGE OF ALL THE DIFFERENCES BETWEEN PREDICTED AND ACTUAL VALUES
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

### AVERAGE OF ALL THE ERRORS SQUARED, TO FIND OUTLIERS
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

### MORE EASILY COMPARABLE TO THE MEAN ABSOLUTE ERROR
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
