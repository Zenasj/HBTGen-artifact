import math
import random
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, activation='softmax', name='class_output')(x)

model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.CategoricalAccuracy()],
             'class_output': [keras.metrics.CategoricalAccuracy()]},
    loss_weights={'score_output': 2., 'class_output': 1.})
callbacks = [
    # keras.callbacks.EarlyStopping("score_output_categorical_accuracy", patience=2, restore_best_weights=True, mode="auto")
    keras.callbacks.EarlyStopping("weighted_categorical_accuracy", patience=2, restore_best_weights=True, mode="auto")
]
import numpy as np
# Generate dummy Numpy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets],
          callbacks=callbacks,
          batch_size=32,
          epochs=10)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
import numpy as np

inp = Input(shape=(1,), dtype=tf.float32)
targets = Input(shape=(1,), dtype=tf.float32)
w = Input(shape=(1,), dtype=tf.float32)
out = Dense(1)(inp)
m = Model(inputs=[inp, w, targets], outputs=out)



def weighted_loss(y_true, y_pred, weights):
      loss = tf.math.squared_difference(y_pred, y_true)
      w_loss = tf.multiply(weights, loss)
      return tf.reduce_mean(tf.reduce_sum(w_loss, axis=-1))
  
m.add_loss(weighted_loss(targets, out, w))

m.compile(loss=None, optimizer='adam')
x = np.ones((512,1))
m.fit([x, x, x], epochs=10)
m.save('test.h5')      
m2 = tf.keras.models.load_model('test.h5')
m2.fit([x, x, x], epochs=10)