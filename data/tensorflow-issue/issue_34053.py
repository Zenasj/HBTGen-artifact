import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

class model(tf.keras.Model):
    def __init__(self, hub_layer):
        super().__init__()
        self.embedding = hub_layer
        self.dense1 = layers.Dense(16, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')
    def call(self, x):
        x = self.embedding(x)
        x = self.dense1(x)
        return self.dense2(x)