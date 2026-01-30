from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras as keras

model = keras.Sequential([
    keras.layers.Dense(4,input_shape=(None,3)),
    keras.layers.Dense(3, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
print(model.input_shape)
print(model.output_shape)

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(4,input_shape=(None,3))
        self.fc2 = tf.keras.layers.Dense(3,activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(2,activation ='softmax')
    def call(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = MLP()
model.build((1,3))
print(model.summary())

print(model.input_shape)
print(model.output_shape)