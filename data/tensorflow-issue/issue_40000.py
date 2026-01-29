# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32) ← Input shape for MobileNet-based classifier

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize MobileNet base model, pretrained on imagenet, without top layers
        self.base = MobileNet(weights="imagenet", include_top=False, input_shape=(224,224,3))
        self.base.trainable = False  # freeze base
        
        # Added classification head as described
        self.gap = GlobalAveragePooling2D()
        self.dense1 = Dense(512, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(5, activation="softmax")  # 5 classes as per tf_flowers dataset

    def call(self, inputs, training=False):
        x = self.base(inputs, training=False)  # base is frozen
        x = self.gap(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel (uncompiled, weights uninitialized)
    # Training and saving happens outside this function
    return MyModel()

def GetInput():
    # Return a random input tensor to test the model (batch size 1) with correct shape and dtype
    # Values normalized roughly between 0 and 1, similar to dataset normalization
    return tf.random.uniform((1, 224, 224, 3), minval=0.0, maxval=1.0, dtype=tf.float32)

