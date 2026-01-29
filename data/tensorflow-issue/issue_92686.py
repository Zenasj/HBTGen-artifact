# tf.random.uniform((1, 10), dtype=tf.float32) ← Input shape inferred from issue: batch size 1, input dimension 10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the simple model architecture from the issue:
        # Input: (batch_size, 10)
        # Hidden layer: Dense(8, ReLU)
        # Output layer: Dense(1, linear)
        self.hidden_layer = tf.keras.layers.Dense(8, activation='relu', name='hidden_layer')
        self.output_layer = tf.keras.layers.Dense(1, activation='linear', name='output_prediction')

    def call(self, inputs):
        # Forward pass
        x = self.hidden_layer(inputs)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel.
    # No pretrained weights specified in the issue, so the model is randomly initialized.
    return MyModel()

def GetInput():
    # The input shape is (1, 10) with dtype=tf.float32, deterministic input from the issue code:
    # input_data = np.arange(10, dtype=np.float32).reshape(1, 10)
    # input_data = (input_data / 10) - 0.5
    import numpy as np
    input_np = (np.arange(10, dtype=np.float32).reshape(1, 10) / 10) - 0.5
    # Convert to tf.Tensor
    return tf.convert_to_tensor(input_np, dtype=tf.float32)

