# tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple sequential-like model with one Dense layer (units=1)
        # This matches the example from the issue with input_shape=(1,)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    """
    Returns an instance of MyModel compiled with SGD optimizer and MSE loss.
    This reflects the typical model construction in the issue.
    """
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


class LoadWeightsCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to load weights and optimizer weights once training begins.

    This is needed because optimizer weights are not created until the first training step.
    Copies weights and optimizer weights into the distributed strategy model after they've been loaded.

    Note:
    - _chief_worker_only = False to run on all replicas, as per issue context.
    """
    _chief_worker_only = False

    def __init__(self, weights, optimizer_weights):
        super().__init__()
        self.weights = weights
        self.optimizer_weights = optimizer_weights

    def on_train_begin(self, logs=None):
        # Set model weights and optimizer weights once training begins,
        # ensuring weights structures are properly initialized.
        self.model.set_weights(self.weights)
        # Safely set optimizer weights if they exist and match expected shapes.
        if self.optimizer_weights:
            try:
                self.model.optimizer.set_weights(self.optimizer_weights)
            except ValueError:
                # This can happen if optimizer weights are not yet initialized.
                # So we skip setting optimizer weights here.
                pass


def GetInput():
    """
    Returns a random batch of inputs compatible with MyModel.
    MyModel expects input shape (batch_size, 1) with float32 dtype,
    mimicking the input_shape=(1,) from the issue examples.
    """
    B = 8  # Batch size chosen arbitrarily
    return tf.random.uniform((B, 1), dtype=tf.float32)

