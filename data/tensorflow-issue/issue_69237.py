from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import pdb
print("TensorFlow version:", tf.__version__)

class network(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=2, activation='relu')

    def call(self, x):
        x = self.conv1(x)

        return x
    
model = network()

inp  = tf.keras.Input(shape=(28, 28, 1))
out = model(inp)
model.build(input_shape=(None, 28, 28, 1))

model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfliteModel = converter.convert()