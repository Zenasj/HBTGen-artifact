# tf.random.uniform((B, 96, 96, 3), dtype=tf.float32) ← Input shape is (batch_size, 96, 96, 3) color images resized from CIFAR-10 (32x32) to 96x96

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Load MobileNetV2 pretrained base without top classification layer,
        # with input shape 96x96x3, output is global average pool features
        self.base_model = keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, pooling='avg')
        # Classification layer for 10 classes
        self.classifier = layers.Dense(10, activation=tf.nn.softmax)
        
    def call(self, inputs, training=False):
        # Forward pass through pretrained base model
        x = self.base_model(inputs, training=training)  # Explicitly pass training to handle batchnorm correctly
        # Final classifier layer output probabilities
        outputs = self.classifier(x)
        return outputs

def my_model_function():
    # Instantiate and return the model instance
    return MyModel()

def GetInput():
    # Generate a random input tensor with batch size 128,
    # matching the data pipeline's batch size and input image shape
    return tf.random.uniform((128, 96, 96, 3), minval=0, maxval=1, dtype=tf.float32)

