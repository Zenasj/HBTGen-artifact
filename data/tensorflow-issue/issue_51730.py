# tf.random.uniform((B, 25), dtype=tf.float32)  # Input shape inferred from model input_dim=25

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers as per the example Dense layers from the issue
        self.dense1 = layers.Dense(64, activation='relu', input_shape=(25,))
        self.dense2 = layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Build the model by calling once with a dummy input (optional but helps with weights init)
    model(tf.zeros((1, 25), dtype=tf.float32))
    return model

def GetInput():
    # Return a random input tensor matching shape (batch_size=1, input_dim=25)
    # Batch size can be arbitrary; here we pick 1
    return tf.random.uniform((1, 25), dtype=tf.float32)

