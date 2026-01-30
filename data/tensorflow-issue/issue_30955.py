import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
TARGET_LAYER_NAME = 'target_layer_name'

# Create the Subclassing API Class
class SubclassedModel(tf.keras.Model):
  def __init__(self, name='subclassed'):
    super(SubclassedModel, self).__init__(name=name)
    self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
    self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name=TARGET_LAYER_NAME)
    self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    self.flatten = tf.keras.layers.Flatten()

    self.dense_1 = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')

  def call(self, inputs, **kwargs):
    x = inputs
    for layer in [self.conv_1, self.conv_2, self.maxpool_1, self.flatten, self.dense_1]:
        x = layer(x)

    return x

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    return tf.TensorShape([shape[0], NUM_CLASSES])

# Initialize a model using the subclassing API
model = SubclassedModel()
model(np.random.random((4, 28, 28, 1)).astype('float32'))  # Sample call to build the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trying to fit some random data, all goes well
training_size = 256
sample_x = np.random.random((training_size, 28, 28, 1)).astype('float32')
sample_y = np.eye(NUM_CLASSES)[np.random.choice(NUM_CLASSES, training_size)]
history = model.fit(sample_x, sample_y, epochs=3, verbose=0)

# Trying to extract a subgraph
submodel = tf.keras.Model([model.inputs], [model.get_layer(TARGET_LAYER_NAME).output])
submodel.summary()

# Trying the same thing with sequential API -- it works

model_seq = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name=TARGET_LAYER_NAME),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
])
model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_seq.fit(sample_x, sample_y, epochs=3, verbose=0)

submodel_seq = tf.keras.Model([model_seq.inputs], [model_seq.get_layer(TARGET_LAYER_NAME).output])
submodel_seq.summary()