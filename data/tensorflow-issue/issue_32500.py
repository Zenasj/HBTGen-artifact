from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

3
import tensorflow as tf

rows = 6
columns = 7

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=[rows * columns, 3]),
  tf.keras.layers.Dense(7, input_shape=[rows * columns * 3]),
])

model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.01),
  loss='mean_squared_error',
  metrics=['accuracy']
)

states = [ [ 1 ] * rows * columns for i in range(20) ]

for iteration in range(1000000):
    print('iteration', iteration)
    model.predict(tf.one_hot(states, dtype='float32', depth=3))