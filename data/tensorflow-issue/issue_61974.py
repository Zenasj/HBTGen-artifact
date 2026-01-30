from tensorflow.keras import layers

# import library
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import BinaryCrossentropy
print(tf.__version__)
# 2.13.0
# import data
X = []
y = []
with open('../../Part1/Week2/data/ex2data1.txt') as file:
# with open('/home/wisdom/vs_code_repository/Python/AndrewNG_ML/Part1/Week2/data/ex2data1.txt', 'r') as file:
    for lines in file:
        colums = lines.strip().split(',')
        X.append([float(colums[0]), float(colums[1])])
        y.append(float(colums[2]))
X = np.array(X)
y = np.array(y)
# normalization
X_mean = np.mean(X, axis=0)
X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
X = (X - X_mean) / (X_max - X_min)

# split data into training_set and testing_set
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]
model = Sequential([Dense(units=32, activation='sigmoid'),
                    Dense(units=16, activation='sigmoid'),
                    Dense(units=1, activation='sigmoid')])
model.compile(loss=BinaryCrossentropy())
model.fit(X_train, y_train, epochs=150) # bug appears this line