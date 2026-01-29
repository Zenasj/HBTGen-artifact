# tf.random.uniform((B, 100), dtype=tf.float32) ← inferred input shape for the initial simple logistic regression example
import tensorflow as tf
import numpy as np
import random
from collections import deque

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Submodel 1: Simple Logistic Regression Model (from the initial memory leak reproduction example)
        self.logistic_input = tf.keras.layers.InputLayer(input_shape=(100,))
        self.flatten = tf.keras.layers.Flatten()
        self.dense_sigmoid = tf.keras.layers.Dense(1, activation='sigmoid')

        # Submodel 2: DQN Model (from TF2.0 memory leak example)
        # We'll replicate the model architecture and replicate the key training logic
        self.dqn_fc1 = tf.keras.layers.Dense(512)
        self.dqn_q_outs = tf.keras.layers.Dense(3)  # Assuming a discrete action space with 3 actions as a placeholder

        # We do not recreate the train loop inside this model here,
        # but will implement a call method to run inference for both models.

    def call(self, inputs, training=False):
        """
        Forward pass outputs a combined dictionary:
        - 'logistic_output': output of the logistic regression submodel
        - 'dqn_output': output of the DQN submodel
        """
        # Input is expected to be a tensor of shape (batch_size, 100) for logistic,
        # and (batch_size, state_dim) for DQN—Here assume same input for simplicity.

        # Logistic regression path
        x1 = self.flatten(self.logistic_input(inputs))
        logistic_out = self.dense_sigmoid(x1)

        # DQN path
        x2 = self.dqn_fc1(inputs)
        dqn_out = self.dqn_q_outs(x2)

        return {'logistic_output': logistic_out, 'dqn_output': dqn_out}


def my_model_function():
    # Returns an instance of MyModel with no extra initialization needed here.
    return MyModel()


def GetInput():
    # We produce a random input tensor that can be fed to the MyModel.call()
    # The model expects float32 inputs with shape (batch_size, 100)
    # We choose batch_size=32 as a reasonable default for testing
    batch_size = 32
    input_shape = (batch_size, 100)
    # Use uniform random floats
    return tf.random.uniform(input_shape, dtype=tf.float32)

