# Data Preprocessing Template

# Importing the libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


# Taking care of missing Data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


# Encoding the chategorical data

# Encode Country Column (independent variable)
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
ct = ColumnTransformer(
    [("Country", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

# Encode Purchased column (dependent variable)
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
