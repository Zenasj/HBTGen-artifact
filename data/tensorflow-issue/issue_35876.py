from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential([
    Dense(10, activation = "relu"),
    Dense(1, activation = "sigmoid")])
model.compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy")
model.fit(
    [[1, 2], [1, 3], [1, 1], [2, 2], [2, 3]],
    [True, False, False, True, True])

print(model.predict([[1, 2], [1, 3], [1, 1]]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

model = Sequential([
    Dense(10, activation = "relu"),
    Dense(2, activation = "softmax")])
model.compile(
    optimizer = "rmsprop",
    loss = "sparse_categorical_crossentropy")

print(model.predict([[1, 2], [1, 3], [1, 1]]))