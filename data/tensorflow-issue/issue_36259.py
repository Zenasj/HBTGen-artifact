# tf.random.uniform((B, 1), dtype=tf.float32) ← Assuming input shape (batch_size, 1) as in the example model input_shape=(1,)

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class CustomLoss(tf.keras.losses.MeanSquaredError):
    # Inherits directly from MeanSquaredError, registering for serialization with no additional changes
    pass

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple dense layer matching original example model
        self.dense = tf.keras.layers.Dense(units=1)

        # Use the registered custom loss instance internally for demonstration
        self.loss_fn = CustomLoss()

    def call(self, inputs, training=False):
        # Forward pass producing predictions
        preds = self.dense(inputs)
        return preds

    def compute_loss(self, inputs, targets):
        # Method to compute loss using the registered CustomLoss
        preds = self.call(inputs)
        return self.loss_fn(targets, preds)

def my_model_function():
    # Return an instance of MyModel, weights uninitialized/randomly initialized
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected input shape of (batch_size, 1)
    # Use batch size B=4 as an example; dtype float32 as typical for dense layers input
    return tf.random.uniform((4, 1), dtype=tf.float32)

