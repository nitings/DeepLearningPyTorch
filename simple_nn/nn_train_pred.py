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

"""
a = torch.empty(5,3)
print(a)
b = torch.zeros(5,3)

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
z = y * y * 3
out = z.mean()

print(z, out)
out.backward()
print(x.grad)
"""

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



mynet = expnet(11, 6, 6, 1)
print(mynet)
criterion = nn.MSELoss()
optimizer = optim.SGD(mynet.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    running_loss = 0.0
    for i in range(np.size(X_train, axis=0)):
        ip = torch.from_numpy(X_train[i:(i+1)]).float()
        target = torch.from_numpy(y_train[i:i+1]).float()
        target = target.view(1, -1)
        optimizer.zero_grad()
        op = mynet(ip)
        loss = criterion(op, target)
        loss.backward()
        optimizer.step()
        
        # Print stats.
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

PATH='./mymodel.pt'
torch.save(mynet.state_dict(), PATH) 

# Load the model.
net = expnet(11,6,6,1)
net.load_state_dict(torch.load('mymodel.pt'))

ip = torch.from_numpy(X_test).float()
op = net(ip)
op = op.detach().numpy()
pred = (op > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

accuracy = ((cm[0][0]+cm[1][1]) / np.size(y_test))*100

print("Accuracy of the model is: ", accuracy)
