# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:17:42 2019

@author: nitings
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Defining the neural net
class expnet(nn.Module):
    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(expnet, self).__init__()
        self.ip_layer = nn.Linear(input_size, h1_size)
        self.h1 = nn.Linear(h1_size, h2_size)
        self.h2 = nn.Linear(h2_size, output_size)
    
    def forward(self, x):
        x = self.ip_layer(x)
        x = F.relu(x)
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = torch.sigmoid(x)
        return x

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Categorical data.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encoding Countries.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding gender variable
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Creating dummy variables only for country.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Removing the first column to avoid dummy variable trap.
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Creating the neural network
mynet = expnet(11, 6, 6, 1)
print("The neural net is: \n")
print(mynet)

# Setting the optimizer, choosing Stochastic Gradient Descent and setting the
# learning rate.
optimizer = optim.SGD(mynet.parameters(), lr=0.01)

# Setting the grad computations to zero.
optimizer.zero_grad()

# Taking one sample data from the training samples just for illustration.
input = torch.from_numpy(X_train[0:1]).float()

# Feeding the input to neural net to calculate the forward value.
output = mynet(input)

# Taking the y train corresponding value for the input above.
target = torch.from_numpy(y_train[0:1]).float()
target = target.view(1, -1)

# Calculating the loss.
criterion = nn.MSELoss()
loss = criterion(output, target)

# Calculating the gradients.
loss.backward()

# Applying the gradient descent.
optimizer.step()
