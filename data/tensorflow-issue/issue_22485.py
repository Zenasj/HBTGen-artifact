# tf.random.uniform((B, 20), dtype=tf.float32) ← The input shape is (batch_size, 20) with float32 dtype

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the shared layers
        self.hidden = Dense(100, activation='relu')
        # Define output layers
        self.out1_layer = Dense(10, activation='relu', name="out1")
        self.out2_layer = Dense(5, activation='relu', name="out2")

    def call(self, inputs, training=None):
        # Forward pass through shared hidden layer
        x = self.hidden(inputs)
        # Compute both outputs
        out1 = self.out1_layer(x)
        out2 = self.out2_layer(x)
        return out1, out2


def zero_loss(y_true, y_pred):
    # Custom zero loss function for output2 to avoid requiring real data when unused
    return tf.constant(0.0)

def my_model_function():
    # Instantiate the model and compile it with:
    # - mse loss for out1 as intended for training
    # - zero_loss for out2 as a workaround to avoid errors in eager execution requiring both outputs' losses
    model = MyModel()
    
    # Build the model by calling it once - helps with summary printing
    dummy_input = tf.random.uniform((1, 20), dtype=tf.float32)
    model(dummy_input)
    
    # Use Adam optimizer (TF 2.x compatible)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model with two outputs, assigning zero loss for out2 to bypass "missing loss" error
    model.compile(
        optimizer=optimizer,
        loss={"out1": "mse", "out2": zero_loss}
    )
    
    return model

def GetInput():
    # Return a dictionary input with key 'input' matching the input name expected in original code
    # We provide a tensor of shape (batch_size=3, 20) with float32 values, matching the original
    X = tf.random.uniform((3, 20), dtype=tf.float32)
    return X

