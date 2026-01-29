# tf.random.uniform((BATCH_SIZE, 224, 224, 3), dtype=tf.float32) ← typical MobileNetV2 input shape

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import MobileNetV2

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 without top, with imagenet weights, input shape 224x224x3
        self.base_model = MobileNetV2(weights="imagenet", include_top=False,
                                     input_tensor=Input(shape=(224, 224, 3)))
        self.base_model.trainable = False
        # Flatten output from base model
        self.flatten = Flatten(name="flatten")
        # Dense layers for bounding box regression head
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(4, activation="sigmoid")  # regression outputs in [0,1]

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and return MyModel with pre-defined MobileNetV2 backbone and bbox head
    return MyModel()

def GetInput():
    # Produce a random uniformly distributed float32 tensor matching model input shape
    # Batch size arbitrarily set to 8, as in the example
    BATCH_SIZE = 8
    return tf.random.uniform((BATCH_SIZE, 224, 224, 3), dtype=tf.float32)

