# tf.random.uniform((10, 2), dtype=tf.float32) ← input shape and dtype inferred from load_data()

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build similar to the provided Sequential model from the issue
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def compute_loss(batch_true, batch_pred):
    # Mean squared error loss, from the issue example
    losses = tf.losses.mean_squared_error(batch_true, batch_pred)
    return tf.reduce_mean(losses)

def train_step(model, batch_input, batch_label, optimizer):
    # Single training step with gradient tape, compiled with tf.function
    with tf.GradientTape() as tape:
        preds = model(batch_input)
        loss = compute_loss(batch_label, preds)
    trainable_variables = model.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    return loss

def train_model(model, data_batches, optimizer, num_epochs=10):
    # Create a tf.function to avoid retracing issues when repeating experiments
    train_step_function = tf.function(train_step)
    for _ in range(num_epochs):
        for batch_input, batch_label in data_batches:
            _ = train_step_function(model, batch_input, batch_label, optimizer)

def GetInput():
    # Generate a random input tensor matching the model input shape (10, 2).
    # Use batch size 10, features 2, dtype float32 as per load_data()
    # Since model expects batches of shape (batch_size, 2)
    return tf.random.uniform((10, 2), dtype=tf.float32)

def my_model_function():
    # Construct MyModel instance and return it.
    # The original example builds optimizer inside run_flow, so not included here.
    return MyModel()

