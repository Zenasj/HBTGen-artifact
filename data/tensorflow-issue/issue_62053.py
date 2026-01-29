# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from dataset used in the issue

import tensorflow as tf
import numpy as np

# Custom optimizer base class selector for TF 2.14 compatibility
def _get_custom_optimizer_parent_class():
    from pkg_resources import parse_version
    if parse_version(tf.__version__) >= parse_version("2.9.0"):
        return tf.keras.optimizers.legacy.Optimizer
    else:
        return tf.keras.optimizers.Optimizer

_custom_optimizer_parent_class = _get_custom_optimizer_parent_class()

class MyModel(tf.keras.Model):
    """
    This model replicates the core Conv2D -> Flatten -> Dense(softmax) architecture
    from the issue. It also includes a custom optimizer that simply assigns gradients
    to variables (for gradient accumulation) and a callback that accumulates gradients.

    The call method defines the forward pass.
    """

    def __init__(self):
        super().__init__()

        # Model layers as described in the issue
        self.conv = tf.keras.layers.Conv2D(3, 3, activation="relu", input_shape=(28, 28, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation="softmax")

        # Custom optimizer instance
        self.custom_optimizer = self._CustomOptimizer()

        # Gradient accumulator callback instance
        self.grad_acc_callback = self._GradAccumulatorCallback()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return self.dense(x)

    class _CustomOptimizer(_custom_optimizer_parent_class):
        """
        A custom optimizer that directly assigns gradients to variables.
        This replicates the minimal custom optimizer in the issue,
        adjusted for TF 2.14 compatibility.
        """

        def __init__(self):
            super().__init__(name="CustomOptimizer")
            # Decorating resource_apply_dense/sparse is likely unnecessary,
            # but following the original code pattern.
            self._resource_apply_dense = tf.function(self._resource_apply_dense)
            self._resource_apply_sparse = tf.function(self._resource_apply_sparse)

        def _resource_apply_dense(self, grad, var):
            # Instead of applying an update, assign var to grad for accumulation
            var.assign(grad)

        def _resource_apply_sparse(self, grad, var, indices):
            # Required to implement for some layers using sparse updates
            pass

        def get_config(self):
            return super().get_config()

    class _GradAccumulatorCallback(tf.keras.callbacks.Callback):
        """
        Callback that accumulates gradients by summing model trainable weights
        after each batch and resets model weights to original to simulate gradient accumulation.
        """

        def set_model(self, model):
            super().set_model(model)
            # Save original weights at start of training
            self.og_weights = model.get_weights()
            # Initialize gradient buffers as zero arrays matching trainable weights
            self.grads = [np.zeros(w.shape, dtype=w.dtype.as_numpy_dtype) for w in model.trainable_weights]

        def on_batch_end(self, batch, logs=None):
            # Accumulate the current trainable weights into grads buffer
            for g, w in zip(self.grads, self.model.trainable_weights):
                g += w.numpy()
            # Reset the model weights to original to avoid updates
            self.model.set_weights(self.og_weights)

        def get_grads(self):
            # Return a copy of accumulated gradients
            return [g.copy() for g in self.grads]

def my_model_function():
    """
    Constructs an instance of MyModel and compiles it
    with the custom optimizer and sparse categorical crossentropy loss,
    matching the usage in the issue.
    """
    model = MyModel()
    # Build the model by calling once with proper input shape tensor
    dummy_input = tf.zeros((1, 28, 28, 1))
    model(dummy_input)

    # Compile the model using the internal custom optimizer
    model.compile(
        optimizer=model.custom_optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def GetInput():
    """
    Returns a tuple of (inputs, labels) compatible with MyModel.
    Inputs are random uint8 in [0,255] cast to float32, shape (100, 28, 28, 1).
    Labels are random integers in [0,9], shape (100,).
    This matches the dataset used in the example.
    """
    x = np.random.randint(0, 256, size=(100, 28, 28, 1), dtype=np.uint8).astype(np.float32)
    y = np.random.randint(0, 10, size=(100,), dtype=np.int32)
    return x, y

