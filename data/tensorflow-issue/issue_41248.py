# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from example: batch with 10 features

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple single Dense layer model, matching the example Dense(4) with input_shape=(10,)
        self.dense = tf.keras.layers.Dense(4)

        # We'll cache a fixed target tensor y to simulate "target_tensors" workaround
        # Initialized as None. Set later via a setter method or property.
        self._fixed_target = None

    def call(self, inputs, training=False):
        # Forward pass: compute model outputs
        return self.dense(inputs)

    def set_fixed_target(self, y):
        # Accept a fixed target tensor to mimic what target_tensors used to do
        self._fixed_target = y

    @tf.function
    def evaluate_with_fixed_target(self, x):
        # Evaluate loss with known fixed target y baked into the function, avoiding passing y each call,
        # mimicking the workaround using a compiled function with embedded target tensor.
        if self._fixed_target is None:
            raise ValueError("Fixed target tensor is not set.")

        y_pred = self.call(x, training=False)
        loss = tf.reduce_mean(tf.square(y_pred - self._fixed_target))
        return loss

def my_model_function():
    # Create and return an instance of MyModel
    model = MyModel()
    return model

def GetInput():
    # Return a random tensor input of shape [batch_size=1, 10 features], dtype float32 as per example
    return tf.random.uniform(shape=(1, 10), dtype=tf.float32)

