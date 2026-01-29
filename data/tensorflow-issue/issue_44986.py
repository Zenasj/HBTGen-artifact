# tf.random.uniform((batch_size, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf
import numpy as np

def to_complex(x):
    return tf.dtypes.complex(x, x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use DenseNet201 base model without top, pretrained on ImageNet
        self.base_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
        # Output converted to complex type via Lambda layer
        self.complex_lambda = tf.keras.layers.Lambda(to_complex)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.complex_lambda(x)
        return x

def my_model_function():
    # Instantiate and compile the model similar to the example
    model = MyModel()
    # Compile with some optimizer and loss although not strictly necessary for prediction
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def GetInput():
    # Match the example input shape used in the issue: large batch of 10000 images, size 224x224, 3 channels
    # But to avoid excessive memory use in practice, we use smaller batch_size here to keep it workable
    # The original example used 10000 which caused OOM in the issue context, so here we pick a smaller batch_size for safety
    batch_size = 4
    input_tensor = tf.random.uniform(
        (batch_size, 224, 224, 3), dtype=tf.float32)
    return input_tensor

