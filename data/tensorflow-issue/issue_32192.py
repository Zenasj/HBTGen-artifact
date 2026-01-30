from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

for k, v in logs.items():
      send[k] = v

for k, v in logs.items():
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, BatchNormalization, MaxPool2D
from tensorflow.keras.optimizers import Adam, Adadelta
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
from tensorflow.keras.callbacks import RemoteMonitor, LambdaCallback, Callback

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(-1,28,28,1) / 255
x_test = x_test.reshape(-1,28,28,1) / 255

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

remote_cb = RemoteMonitor(root="http://localhost:9000", path="/publish/epoch/end/", send_as_json=True)

model = Sequential()
model.add(Conv2D(64,(3,3), activation="relu", input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=Adam(),
              metrics = ["accuracy"])

model.fit(x_train[:3000], y_train[:3000], epochs=5, batch_size=64, callbacks=[remote_cb])

for k, v in logs.items():
      send[k] = v

for k, v in logs.items():
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v