# https://medium.com/diogo-menezes-borges/predicting-banks-churn-with-artificial-neural-networks-f48393fb1f9c

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X_orig = dataset.iloc[:, 3:13].values
y_orig = dataset.iloc[:, 13].values

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() # encode 'country' category feature
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # avoiding 'dummy variable trap'

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# thumbrule to choose number of nodes = avg of number of nodes in input layer + output layer: (11 + 1)/2 = 6
# 'uniform' - initializes the starting weights according to an "uniform" distribution
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# our output is a binary outcome so one node would suffice 
# activation func would be 'sigmoid' to provide the probability of one of the binary outcomes
# for more than 2 categories then "units = 3" and "activation = 'softmax'"
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# optimizer - algorithm used to optimized the weights ('adam' is one the efficient SGD algos)
# loss - loss or the cost function between y and y-hat)
#       "categorical_crossentropy" - for more than 2 categories
# metrics - metric(s) used to evaluate the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Homework: prediction for a single customer with these features -
'''
Geography : France

Credit Score: 600

Gender: Male

Age: 40

Tenure: 3

Balance: 60000

Number of Products: 2

Has Credit Card: Yes

Is Active Member: Yes

Estimated Salary: 50000
'''

