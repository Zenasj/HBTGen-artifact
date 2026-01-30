import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_layer = tf.keras.Input(shape=(BOARD_SIZE * 3,))
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(input_layer)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 100, activation='relu')(x)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(x)
q_values = tf.keras.layers.Dense(BOARD_SIZE, activation=None, name='q_values')(x)
probabilities = tf.keras.layers.Softmax(name='probabilities')(q_values)

self.model = tf.keras.Model(inputs=input_layer, outputs=[probabilities, q_values])
if run_tf_function:
    self.model.compile(optimizer='adam', loss = [None, tf.keras.losses.MeanSquaredError()])
else:
    self.model.compile(optimizer='adam', loss = [None, tf.keras.losses.MeanSquaredError()], experimental_run_tf_function = False)

input_layer = tf.keras.Input(shape=(BOARD_SIZE * 3,))
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(input_layer)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 100, activation='relu')(x)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(x)
q_values = tf.keras.layers.Dense(BOARD_SIZE, activation=None, name='q_values')(x)
probabilities = tf.keras.layers.Softmax(name='probabilities')(q_values)

self.model = tf.keras.Model(inputs=input_layer, outputs=[probabilities, q_values])
self.model.compile(optimizer='adam', loss = [None, tf.keras.losses.MeanSquaredError()])

res = self.model.fit(np_inputs, {'q_values': np_targets}, verbose=0)

input_layer = tf.keras.Input(shape=(BOARD_SIZE * 3,))
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(input_layer)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 100, activation='relu')(x)
x = tf.keras.layers.Dense(BOARD_SIZE * 3 * 9, activation='relu')(x)
q_values = tf.keras.layers.Dense(BOARD_SIZE, activation=None, name='q_values')(x)
probabilities = tf.keras.layers.Softmax(name='probabilities')(q_values)

self.model = tf.keras.Model(inputs=input_layer, outputs=[q_values, probabilities])
self.model.compile(optimizer='adam', loss = [tf.keras.losses.MeanSquaredError(), None])