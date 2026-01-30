import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=32, output_dim=64, input_length=32),
    tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.GRU(128, return_sequences=True)),
    tf.keras.layers.Dense(32, activation='softmax')
])