import random
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import model_from_json
from tensorflow.keras import Input, Model, regularizers, layers

tf.compat.v1.disable_eager_execution()


inputs = Input(shape=(10,))
d = layers.Dense(10, kernel_initializer='ones')
x = d(inputs)
outputs = layers.Dense(1)(x)
model = Model(inputs, outputs)
  
model.add_loss(tf.reduce_mean(outputs))

reg_losses = [
    tf.cast(tf.size(input=w), tf.float32) for w in model.trainable_weights
    ]
model.add_loss(tf.add_n(reg_losses))

model.summary()

model.compile(optimizer="adam", loss=[None] * len(model.outputs))
model.fit(np.random.random((2, 10)))

model_json = model.to_json()
json_filename = "model.json"
with open(json_filename, "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

json_file = open(json_filename, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)  # UnboundLocalError: local variable 'kwargs' referenced before assignment
print("model_from_json ok")