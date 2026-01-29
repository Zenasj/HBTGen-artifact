# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape B=batch size, 1 feature per example

import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the trainable model (the main model) as per original snippet
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')

        # Define the already trained model (model_weight) used to compute weight
        # For demonstration, we create a simple sub-model here.
        # In practice, this would be loaded weights or a separate trained model.
        self.model_weight = self._build_model_weight()

    def _build_model_weight(self):
        # The weight model expects inputs of shape (?, 2)
        inputs = layers.Input(shape=(2,))
        x = layers.Dense(32, activation='relu')(inputs)
        x = layers.Dense(16, activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=out)
        # For demonstration, weights are random initialized
        return model

    def call(self, inputs, training=False):
        """
        Forward pass outputs:
          - predictions: output of main model (shape batch x 1)
          - weights: computed using model_weight on combined input and theta0
          - loss_weighted_diff: weighted component for loss calculation returned for info (optional)
          
        This setup fuses two models:
          1) main model receiving inputs,
          2) model_weight evaluating input combined with theta0 stack.

        Note: The loss function incorporating weights w is to be implemented separately 
        (e.g., using a custom training loop or a tf.keras loss wrapper passing these).
        """
        # Main model forward path
        x = tf.cast(inputs, tf.float32)  # ensure float32 dtype
        x1 = self.dense1(x)
        x2 = self.dense2(x1)
        predictions = self.output_layer(x2)  # shape (B,1)

        # Prepare data for model_weight that expects 2-element vectors per sample:
        # For demonstration, we use theta0 = 0 for all batch elements (can be a parameter)
        batch_size = tf.shape(inputs)[0]
        theta0_stack = tf.zeros_like(inputs, dtype=tf.float32)  # shape (B, 1)

        # Concatenate x and theta0_stack along last axis, shape (B, 2)
        data_for_weight = tf.concat([x, theta0_stack], axis=-1)

        # Compute weights via model_weight (output shape (B,1))
        # Use call instead of predict to maintain graph and tensor compatibility
        w = self.model_weight(data_for_weight, training=False)

        # Note: This forward pass returns the 3 results for transparent usage
        return predictions, w

def my_loss_wrapper(theta=0.0):
    """
    Returns a tf.keras compatible loss function that incorporates the secondary model_weight 
    output as weights for the loss.
    Assumption: theta is a scalar (float) controlling the second input to model_weight.

    Since Keras loss functions get (y_true, y_pred) only,
    this implements a closure approach using a custom training loop in practice.
    For demonstration, we define a 'loss' that expects a tuple y_pred with (predictions, w).
    """
    def loss_fn(y_true, y_pred):
        # y_pred expected as a tuple (predictions, w)
        predictions, w = y_pred
        y_true = tf.cast(y_true, tf.float32)
        # Compute the weighted loss (mean over batch)
        # Original loss: mean(y_true*(y_true - pred)^2 + (w)^2*(1 - y_true)*(y_true - pred)^2)
        squared_diff = tf.square(y_true - predictions)
        weighted_loss = y_true * squared_diff + tf.square(w) * (1.0 - y_true) * squared_diff
        return tf.reduce_mean(weighted_loss)
    return loss_fn

def my_model_function():
    # Return an instance of MyModel, including the built sub-models.
    return MyModel()

def GetInput():
    # Return random input tensor matching shape expected by MyModel: (batch, 1)
    batch_size = 500  # inferred from original issue typical batch size
    return tf.random.uniform((batch_size, 1), minval=0, maxval=10, dtype=tf.float32)

