# tf.random.normal((B, 32, 32, 1), dtype=tf.float32) ← Inferred input shape from original example

import tensorflow as tf

class DataDepInit(tf.keras.layers.Layer):
    """A layer with data-dependent initialization across multiple replicas with aggregation.

    This layer initializes its trainable weight (`self.w`) on the first batch of input data,
    computing the mean over spatial and batch dims, aggregated across replicas by MEAN.

    After initialization, the subtraction by `self.w` is performed on inputs.
    """
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        # Weight shape = (1, 1, 1, channels)
        channels = input_shape[-1]
        self.w = self.add_weight(
            name="mean",
            shape=(1, 1, 1, channels),
            dtype=tf.float32,
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,  # Aggregate across replicas
            initializer=tf.zeros_initializer()
        )
        self.initialized = self.add_weight(
            name="init",
            shape=(),  # scalar bool tensor
            dtype=tf.bool,
            trainable=False,
            initializer=tf.zeros_initializer()
        )
        # Mark layer built
        super().build(input_shape)

    def initialize(self, x):
        # Compute mean over [batch, height, width], keep channel dim and batch dims for assignment
        # Use reduce_mean to get per-channel mean across batch and spatial dims
        mean = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True)
        # Assign mean to self.w, aggregated across replicas
        self.w.assign(mean)

    def call(self, x):
        # Initialize weights on first call with data-dependent init
        if not self.initialized:
            # To avoid retracing issues and sync errors with assign in distributed context,
            # do assignment outside tf.function or use workaround (here assumes eager execution).
            self.initialize(x)
            self.initialized.assign(True)
        return x - self.w


class MovingNorm(tf.keras.layers.Layer):
    """Example layer implementing moving mean and variance with distributed aggregation.

    This is from the working snippet in the issue and shows usage of aggregation and assign
    in call with distributed strategy.
    """
    def __init__(self, mean: float = 0.0, var: float = 1.0):
        super().__init__()
        self.target_mean_init = mean
        self.target_var_init = var

    def build(self, input_shape):
        shape = (1, 1, 1, input_shape[-1])
        # moving_mean and moving_var are non-trainable, aggregated by MEAN for sync
        self.moving_mean = self.add_weight(
            'moving_mean', shape,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN)
        self.moving_var = self.add_weight(
            'moving_var', shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN)
        # target_mean and target_var are trainable parameters
        self.target_mean = self.add_weight(
            'target_mean', shape,
            initializer=tf.keras.initializers.Constant(self.target_mean_init),
            trainable=True)
        self.target_var = self.add_weight(
            'target_var', shape,
            initializer=tf.keras.initializers.Constant(self.target_var_init),
            trainable=True)

        super().build(input_shape)

    def call(self, x, decay: float):
        # Update moving statistics when decay < 1.0 (training)
        if decay < 1.0:
            mean, var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
            var = tf.maximum(var, 1e-3)
            # Assign moving mean and var with distributed mean aggregation
            self.moving_mean.assign(decay * self.moving_mean + (1 - decay) * mean)
            self.moving_var.assign(decay * self.moving_var + (1 - decay) * tf.sqrt(var))
        # Normalize output with target parameters and moving statistics
        mult = self.target_var / self.moving_var
        x_norm = mult * x + (self.target_mean - self.moving_mean * mult)
        return x_norm


class MyModel(tf.keras.Model):
    """Fused model that encapsulates DataDepInit and MovingNorm layers.

    This model uses the data-dependent initialization layer first on inputs,
    then applies the moving norm layer with a fixed decay for demonstration.

    The call returns the difference (boolean) between DataDepInit output and
    MovingNorm output within a small epsilon tolerance, as an example comparison.
    """
    def __init__(self):
        super().__init__()
        self.data_dep_init = DataDepInit()
        self.moving_norm = MovingNorm()

    def call(self, x):
        # Apply data-dependent init layer
        ddi_out = self.data_dep_init(x)
        # Apply moving norm with decay = 0.9 (as example)
        mn_out = self.moving_norm(x, decay=0.9)

        # As per requirement to fuse models and compare:
        # Compute absolute diff and check if approx equal
        diff = tf.abs(ddi_out - mn_out)
        tol = 1e-5
        comparison = tf.reduce_all(diff < tol)

        # For demonstration, output a dict with layer outputs and comparison result
        # Alternatively, return comparison as boolean tensor cast to int32 for tf.function compatibility
        return {"ddi_out": ddi_out, "moving_norm_out": mn_out, "within_tolerance": tf.cast(comparison, tf.int32)}


def my_model_function():
    # Instantiate MyModel
    return MyModel()


def GetInput():
    # Return a batch of input matching expected shape (e.g. batch size 8 for testing)
    batch_size = 8
    height, width, channels = 32, 32, 1
    # Generate random normal input tensor
    return tf.random.normal(shape=(batch_size, height, width, channels), dtype=tf.float32)

