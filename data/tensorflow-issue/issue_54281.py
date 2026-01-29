# tf.random.uniform((B, 32), dtype=tf.float32) ← the input shape is (batch_size, 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model layers: single dense layer with 1 output unit
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

    def train_step(self, data):
        # Custom train_step implementation that returns the results of the metrics
        # This matches the issue's minimal example
        # Note: This custom train_step doesn't actually perform training,
        # just returns metric results, illustrating the problem described.
        # A realistic custom train_step would include gradient computation and optimizer steps.
        x, y = data
        y_pred = self(x, training=True)

        # Compute the loss using the compiled loss function
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Update metrics (includes the metric objects passed during compile())
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dictionary mapping metric names to current values
        return {m.name: m.result() for m in self.metrics}


def my_model_function():
    # Create inputs with shape (None, 32)
    inputs = tf.keras.Input(shape=(32,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = MyModel()
    # Build the model by calling once or using inputs/outputs signature
    model.__call__(tf.zeros((1, 32)))
    # Compile with optimizer, loss, metrics as in the issue example
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def GetInput():
    # Provide a batch of inputs with shape (1000, 32) matching example data
    # dtype float32 matches typical TensorFlow float inputs
    return tf.random.uniform((1000, 32), dtype=tf.float32)

