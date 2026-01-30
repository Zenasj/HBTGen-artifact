from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D


(X_train, y_train), _ = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0


with tf.device('cpu:0'):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(32, 32, 3)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    
model.fit(X_train, y_train,
          batch_size=1024,
          epochs=2)