# tf.random.uniform((16, 1000, 3), dtype=tf.float32) ← input shape and dtype inferred from example data

import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Print to check if running eagerly (True when eager)
        print("Loss running eagerly:", tf.executing_eagerly())
        x = y_true + y_pred
        return tf.reduce_mean(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Dense layer applied across the last dimension of inputs with shape (1000, 3)
        self.dense = tf.keras.layers.Dense(3)

        # Use CustomLoss internally here for demonstration
        self.loss_fn = CustomLoss()

    def call(self, inputs, training=None):
        outputs = self.dense(inputs)
        # Typically loss is not called inside call; the framework does this. But provide output only.
        return outputs

    def compute_loss(self, y_true, y_pred):
        # Provide a method to compute custom loss outside the call (runtime)
        return self.loss_fn(y_true, y_pred)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching input shape (16 batch, 1000 sequence, 3 features), float32
    return tf.random.uniform(shape=(16, 1000, 3), dtype=tf.float32)

