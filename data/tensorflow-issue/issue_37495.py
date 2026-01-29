# tf.random.uniform((B, 6), dtype=tf.float32) ← Input shape inferred from data_X (100,6)

import tensorflow as tf
import tensorflow_probability as tfp

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model layers based on the issue's architecture:
        # Input: shape=(6,)
        # Hidden Layer: Dense 5000 units with ReLU, He normal initializer, L1 regularizer
        # Dropout 0.3
        # Output: Dense 1 with sigmoid activation
        
        self.dense_input = tf.keras.layers.Dense(
            6, activation='linear', name='input_layer'
        )
        self.hidden = tf.keras.layers.Dense(
            5000,
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l1(1e-3),
            name='hidden_layer'
        )
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')

    def call(self, inputs, training=False):
        # Forward pass through layers
        x = self.dense_input(inputs)
        x = self.hidden(x)
        x = self.dropout(x, training=training)
        out = self.output_layer(x)
        return out

def my_model_function():
    # Returns an instance of MyModel, weights uninitialized since no pretrained provided
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (batch, 6 features)
    # Batch size is arbitrary, choose 32 as in the issue's example
    return tf.random.uniform((32, 6), dtype=tf.float32)

# Custom Pearson correlation as a loss must be implemented outside the model class
# but here we only build the model and input generator as requested.

# Notes/Assumptions:
# - Input shape (batch, 6) inferred from data_X shape in the issue.
# - Model closely follows the stated architecture in the issue code snippet.
# - Dropout is set to training mode only during training.
# - We do NOT implement the custom loss here since the instruction is to provide model + input.
# - If needed, the Pearson correlation loss is defined in the issue but is external to the model.
#
# This model code is compatible with TF 2.20.0 XLA compilation requirements.

