import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

user_model = tf.keras.Sequential([
    tf.keras.Input((1,), dtype='int64'),
    tf.keras.layers.Embedding(20000000, 10)
])

user_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.2),loss='mse')