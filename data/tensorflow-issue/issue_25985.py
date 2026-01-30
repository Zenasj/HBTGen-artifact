from tensorflow import keras
from tensorflow.keras import layers

py
import tensorflow as tf

# import pdb; pdb.set_trace()
inputs = tf.keras.layers.Input(batch_shape=(1, 1, 1))

state_h = tf.keras.layers.Input(batch_shape=(1, 1))
state_c = tf.keras.layers.Input(batch_shape=(1, 1))

states = [state_h, state_c]

decoder_out = tf.keras.layers.LSTM(1, stateful=True)(
    inputs,
    initial_state=states
)

model = tf.keras.Model([inputs, state_h, state_c], decoder_out)
model.reset_states()