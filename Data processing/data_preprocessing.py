# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# matrix of feature
X = dataset.iloc[:, :-1].values
# depedent variable vector
Y = dataset.iloc[:, -1].values


# Taking care of missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


# Encoding the chategorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling 
from sklearn.preprocessing import  StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
