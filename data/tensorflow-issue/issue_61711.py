import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

model = tf.keras.MLP(
    hidden_channels=[128, 64, 32, 10],
    norm_layer=tf.keras.layers.BatchNormalization,
    activation_layer=tf.keras.layers.ReLU,
    dropout=0.2,
)

class MLP(tf.keras.Sequential):
    def __init__(self, hidden_channels, norm_layer, activation_layer, dropout, **kwargs):
        super().__init__(**kwargs)
        for units in hidden_channels[:-1]:
            self.add(tf.keras.layers.Dense(units))
            self.add(norm_layer())
            self.add(activation_layer())
            self.add(tf.keras.layers.Dropout(dropout))
        self.add(tf.keras.layers.Dense(hidden_channels[-1]))
        self.add(tf.keras.layers.Dropout(dropout))