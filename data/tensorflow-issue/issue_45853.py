# tf.random.uniform((32, 400, 16), dtype=tf.float32) ← inferred input shape from the test/data discussion in the issue

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a small CNN model as discussed in the issue
        self.conv1 = tf.keras.layers.Conv1D(128, 400, strides=4, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

        # Define a medium model with Bidirectional LSTM + Conv1D as per issue snippet
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(512, activation='relu', return_sequences=True)
        )
        self.lstm = tf.keras.layers.LSTM(512, activation='relu', return_sequences=True)
        self.conv2 = tf.keras.layers.Conv1D(128, 400, strides=4, padding='same')
        self.flatten2 = tf.keras.layers.Flatten()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2_1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2_2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2_3 = tf.keras.layers.Dense(64, activation='relu')
        self.out2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # We will run the input through both models and compare outputs
        # This fusion reflects the comments where both small and medium models were benchmarked
        
        # Small model forward
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        out_small = self.out(x)

        # Medium model forward
        y = self.bilstm(inputs)
        y = self.lstm(y)
        y = self.conv2(y)
        y = self.flatten2(y)
        y = self.dropout2(y, training=training)
        y = self.dense2_1(y)
        y = self.dense2_2(y)
        y = self.dense2_3(y)
        out_medium = self.out2(y)

        # Compare outputs: for example, absolute difference
        diff = tf.abs(out_small - out_medium)
        # Optionally return boolean if outputs are close within tolerance
        comparison = tf.less_equal(diff, 1e-3)

        # Return dictionary of outputs + comparison to illustrate fusion and comparison logic
        return {
            'output_small': out_small,
            'output_medium': out_medium,
            'difference': diff,
            'close_enough': comparison
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape matching model input:
    # batch_shape: (32, 400, 16) - based on batch_shape in the issue's benchmarking code
    return tf.random.uniform((32, 400, 16), dtype=tf.float32)

