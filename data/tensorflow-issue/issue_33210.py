import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import SGD

num_features = 100
train_x = np.random.rand(40, num_features)
train_y = np.random.randint(2, size=40)

# The input layer
input_layer = Input(shape=(num_features,), name="Input")
output = Dense(10, activation='sigmoid', name="Hidden_1")(input_layer)
output = Dense(1, activation='sigmoid', name="Output")(output)
model = Model(inputs=input_layer, outputs=output)

sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
                     optimizer=sgd,
                      metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("out_dir", datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=2, write_graph=True, write_images=True)
my_callbacks = [tensorboard_callback]

model.fit(x=train_x, y=train_y,
                  validation_split=.2,
                  epochs=5,
                  callbacks=my_callbacks)