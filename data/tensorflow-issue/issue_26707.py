import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)
config = model.get_config()
new_model = tf.keras.Model.from_config(config)

import tensorflow as tf
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

inputs = tf.keras.Input(shape=(32,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)
config = model.get_config()
new_model = tf.keras.Model.from_config(config)

import tensorflow as tf
import numpy as np

print("TensorFlow " + tf.__version__)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(32,)),
        tf.keras.layers.Dense(128, activation='relu',),
          tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
            ])

# The compile step specifies the training configuration
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                      loss='categorical_crossentropy',
                                    metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)
# json approach
json_config = model.to_json()
new_model = tf.keras.models.model_from_json(json_config)
# model cloning
cloned_model=tf.keras.models.clone_model(model)
# config approach
dict_config = model.get_config()
new_model2 = tf.keras.models.model_from_config(dict_config)

model.summary()
new_model.summary()
cloned_model.summary()
new_model2.summary()