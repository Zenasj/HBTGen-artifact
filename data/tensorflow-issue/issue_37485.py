import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, activation = 'relu', input_shape = (timesteps, features), return_sequences = True),
        tf.keras.layers.LSTM(16, activation = 'relu', return_sequences = False),
        tf.keras.layers.RepeatVector(timesteps),
        tf.keras.layers.LSTM(16, activation = 'relu', return_sequences = True),
        tf.keras.layers.LSTM(32, activation = 'relu', return_sequences = True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation = 'softmax'))
    ])

    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ['accuracy'])