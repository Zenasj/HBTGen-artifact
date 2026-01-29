# tf.random.uniform((B, input_dim), dtype=tf.float32) ← inferred input shape, batch size B, input_dim as model input dimension

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, input_dim=20, output_dim=10, num_models=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_models = num_models

        # Ensembles of Sequential models for mean predictions and variance predictions
        self.mean = [
            tf.keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.output_dim)
            ]) for _ in range(self.num_models)
        ]
        self.variance = [
            tf.keras.Sequential([
                layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.output_dim)
            ]) for _ in range(self.num_models)
        ]

    def call(self, inputs, training=None, mask=None):
        """
        Forward pass calls each ensemble member on the inputs,
        passing the training flag so layers like Dropout behave properly.

        Returns:
          A tuple of stacked mean predictions and stacked variance predictions:
          shapes:  (num_models, batch_size, output_dim)
        """
        mean_predictions = []
        variance_predictions = []
        for idx in range(self.num_models):
            # Pass the training flag to each sub-model
            mean_predictions.append(self.mean[idx](inputs, training=training))
            variance_predictions.append(self.variance[idx](inputs, training=training))

        mean_stack = tf.stack(mean_predictions)       # shape: (num_models, batch_size, output_dim)
        variance_stack = tf.stack(variance_predictions)

        return mean_stack, variance_stack


def my_model_function():
    # Create instance with example dims; these can be adjusted as needed
    return MyModel(input_dim=20, output_dim=10, num_models=3)


def GetInput():
    # Generate a batch of random inputs compatible with MyModel
    # Batch size is arbitrary; choose 4 here
    batch_size = 4
    input_dim = 20  # Must match MyModel's expected input_dim

    # Random input tensor with floats between 0 and 1
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

