# tf.random.uniform((B, 6400), dtype=tf.float32) ← Input shape inferred from Keras Input(shape=(6400,)) in the reproduced minimal code

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Setting mixed precision policy to 'mixed_float16'
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        num_units = 8192
        
        self.dense_1 = layers.Dense(num_units, activation='relu', name='dense_1')
        self.dense_2 = layers.Dense(num_units, activation='relu', name='dense_2')
        # Output layer with 1000 units, no activation here, will apply ReLU separately
        self.dense_logits = layers.Dense(1000, name='dense_logits')
        # Final ReLU activation on logits with output cast to float32
        self.predictions = layers.Activation('relu', dtype='float32', name='predictions')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_logits(x)
        x = self.predictions(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Mimic input tensor based on input shape (6400,) with dtype float32,
    # batch size chosen as 8192 (as example batch size in issue).
    batch_size = 8192
    input_shape = (batch_size, 6400)
    # Values sampled from uniform distribution [0,1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

