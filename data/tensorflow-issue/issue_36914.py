# tf.random.uniform((B, n_timestep, 6), dtype=tf.float32)  # Assumed input shape for Conv1D model from issue

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense

class MyModel(tf.keras.Model):
    def __init__(self, n_timestep=10, num_classes=3):
        """
        Reconstructed model based on the description in the issue:
        - 2 Conv1D layers with 32 filters, kernel size 3, relu activation
        - Dropout 0.5
        - Flatten
        - Dense 100 relu
        - Dense num_classes softmax
        
        The input_shape is (n_timestep, 6) with 6 channels/features per timestep,
        consistent with the user's input shape in the issues.
        
        The issue suggests MaxPooling1D causes problems in TFLite micro on ESP32,
        so this version excludes pooling, which they confirmed works.
        """
        super().__init__()
        self.model = Sequential([
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timestep, 6)),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Dropout(0.5),
            # MaxPooling1D(pool_size=2),  # Omitted due to issues in microcontroller inference
            Flatten(),
            Dense(100, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        # Delegate call to internal Keras Sequential model
        return self.model(inputs, training=training)


def my_model_function():
    # Return an instance with default timestep=10 and 3 classes as per example
    return MyModel(n_timestep=10, num_classes=3)


def GetInput():
    # Generate a random input tensor matching (batch_size=1, n_timestep=10, features=6)
    # Using float32 as input dtype since the TFLite example uses float inputs for compatibility testing
    n_timestep = 10
    batch_size = 1
    input_channels = 6
    return tf.random.uniform((batch_size, n_timestep, input_channels), dtype=tf.float32)

