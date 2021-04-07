# Simple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)

# Feature scaling
'''from sklearn.preprocessing import  StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])'''

# Linear Regression
reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)


# graph
plt.scatter(X_test, y_test, color='red' )
plt.plot(X_train, reg.predict(X_train) , color='blue')
plt.title('Salary vs Experience(traning set)')
plt.xlabel('Experience in year')
plt.ylabel('Salary')
plt.show()
