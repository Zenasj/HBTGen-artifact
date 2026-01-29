# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape inferred from keras.Input(shape=(32,)) in the code

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Moving Variables _outside_ TPU strategy scope is recommended for TPU usage,
        # because TPU does not support variables with unknown/dynamic shapes on device.
        # Here, we keep them on CPU/host for safe assignment.
        # Shapes are [None, 32], [None], [None, 1] respectively, but dynamic dimension -> workaround.
        
        # To emulate the behavior, we create variables on CPU with unknown first dimension,
        # dtype float32 to avoid TPU bfloat16 issues for these variables which are used only for storage.
        # Using tf.Variable shape=[0, ...], and allow shape updates via assign with concat.
        
        # Use a non-distributed variable placed on CPU explicitly:
        with tf.device('/CPU:0'):
            self.val_x = tf.Variable(
                initial_value=tf.zeros((0, 32), dtype=tf.float32),
                trainable=False,
                shape=tf.TensorShape([None, 32])
            )
            self.val_gt = tf.Variable(
                initial_value=tf.zeros((0,), dtype=tf.float32),
                trainable=False,
                shape=tf.TensorShape([None])
            )
            self.val_pred = tf.Variable(
                initial_value=tf.zeros((0,1), dtype=tf.float32),
                trainable=False,
                shape=tf.TensorShape([None, 1])
            )

        # Define a simple Dense layer for the forward pass
        self.dense = keras.layers.Dense(1, dtype='float32')

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

    def test_step(self, data):
        # Custom test step copying the original code logic to collect batch data

        x, y = data
        y_pred = self(x, training=False)
        # Compute loss and update metrics as usual
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        # Attempt to append batch data to val_x, val_gt, val_pred variables dynamically
        # Use tf.concat and assign to variables placed on CPU device.
        # This triggers the original TPU error if variables placed on TPU device.
        # Here, ensuring variables placed on CPU and concatenation happens on CPU.

        with tf.device('/CPU:0'):
            # Concat along axis=0 the existing contents with current batch tensors.
            self.val_x.assign(tf.concat([self.val_x, x], axis=0))
            self.val_gt.assign(tf.concat([self.val_gt, y], axis=0))
            self.val_pred.assign(tf.concat([self.val_pred, y_pred], axis=0))

        # Return dictionary of metric results
        return {m.name: m.result() for m in self.metrics}


def my_model_function():
    # Build and compile an instance of MyModel for usage
    
    # Create model inputs tensor
    inputs = keras.Input(shape=(32,))
    # Instantiate MyModel with inputs and outputs to match Keras Model subclass usage.
    # But since MyModel inherits from tf.keras.Model directly, we build it standalone.
    # We wrap the Dense layer inside MyModel, so inputs are passed through MyModel instance.
    model = MyModel()
    # Build the model by calling on dummy input
    _ = model(tf.zeros((1, 32), dtype=tf.float32))
    # Compile with optimizer, loss, and metrics as per the original snippet
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    return model


def GetInput():
    # Return a random input tensor matching the expected input shape of MyModel forward pass
    # As per keras.Input(shape=(32,)) this is (batch_size, 32)
    # Choose batch_size=4 arbitrarily to match usage in model.fit(batch_size=4)
    return tf.random.uniform((4, 32), dtype=tf.float32)

