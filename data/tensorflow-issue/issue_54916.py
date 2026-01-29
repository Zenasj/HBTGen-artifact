# tf.random.uniform((B, 559, 331, 3), dtype=tf.float32) ← input shape inferred from the code snippet and data generator

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Base EfficientNetB2 model without top, inputs fixed to (559, 331, 3)
        self.baseModel = EfficientNetB2(
            weights=None,
            include_top=False,
            input_tensor=Input(shape=(559, 331, 3))
        )
        # Head layers as described
        self.avg_pool = AveragePooling2D(pool_size=(7, 7))
        self.flatten = Flatten(name="flatten")
        self.dense1 = Dense(512, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(5, activation="softmax")

        # Freeze base model layers
        for layer in self.baseModel.layers:
            layer.trainable = False

    def call(self, inputs, training=False):
        x = self.baseModel(inputs, training=training)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (batch, 559, 331, 3)
    # Batch size is chosen as 4 here as a reasonable default
    batch_size = 4
    # Using tf.random.uniform with values between 0 and 1 to mimic normalized image inputs
    return tf.random.uniform(shape=(batch_size, 559, 331, 3), dtype=tf.float32)

