# tf.random.uniform((B, 96, 96, 1), dtype=tf.float32)  # Inferred input shape from MobileNet adaptation in issue

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, Softmax

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Adapted MobileNetV1 model from issue comments, designed to work on embedded targets
        # with single-channel input and smaller alpha for a lightweight model.
        base_model = MobileNet(
            include_top=False,
            weights=None,
            input_shape=(96, 96, 1),
            pooling='avg',  # Global average pooling to reduce spatial dims
            alpha=0.25,
            dropout=0.001
        )
        self.base_model = base_model
        # Top layers adapted to match a final classification for 2 classes (like binary classification)
        self.dropout = Dropout(0.001)
        self.dense = Dense(2)
        self.softmax = Softmax()

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel with no pretrained weights
    model = MyModel()
    return model

def GetInput():
    # Return a random float32 tensor shaped (1, 96, 96, 1) matching the model input
    # Using batch size 1 as typical for embedded inference
    return tf.random.uniform((1, 96, 96, 1), minval=0, maxval=1, dtype=tf.float32)

