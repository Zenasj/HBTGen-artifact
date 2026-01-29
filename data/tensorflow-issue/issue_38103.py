# tf.random.uniform((B, 10), dtype=tf.float32)  <-- Input shape inferred from example: (batch_size, 10)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.eager import backprop

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model layers
        self.dense = layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

    # Custom train_step with tf.function and input_signature for serialization
    @tf.function(input_signature=[tuple((
        [tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
         tf.TensorSpec(shape=(None,), dtype=tf.int64)]
    ))])
    def train_step(self, data):
        # data is a tuple (x, y)
        x, y = data
        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Compute gradients and update weights
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # Update metrics (if any)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        base_config = super().get_config()
        # Including a reference to train_step decorated function for potential serialization
        base_config.update({'train_step': tf.function(self.train_step)})
        return base_config

def my_model_function():
    # Instantiate and compile the model with suitable loss and optimizer
    model = MyModel()
    # Loss and optimizer chosen to align with train_step example (MSE and SGD)
    model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
    return model

def GetInput():
    # Return a tuple of input tensors matching the train_step input_signature:
    # - features: (batch_size, 10) float32 tensor
    # - labels: (batch_size,) int64 tensor (class indices or integer labels)
    B = 4  # batch size chosen arbitrarily for demonstration

    x = tf.random.uniform((B, 10), dtype=tf.float32)
    y = tf.random.uniform((B,), minval=0, maxval=2, dtype=tf.int64)  # dummy integer labels 0 or 1

    return (x, y)

