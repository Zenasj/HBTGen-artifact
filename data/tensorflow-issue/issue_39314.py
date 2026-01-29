# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape based on example usage in issue (10 features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates the minimal example from the issue describing
    a custom Count metric that tracks total samples processed during training.
    The main model is a simple single Dense layer as in the example.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # Simple Dense layer as in the example
        self.dense = tf.keras.layers.Dense(1)

        # Custom Count metric to count total examples seen during training
        self.count_metric = Count(name='counter')

    def compile(self, optimizer, loss):
        # Compile method to accept needed optimizer and loss
        super(MyModel, self).compile(optimizer=optimizer, loss=loss, metrics=[self.count_metric])

    def call(self, inputs, training=False):
        # Forward pass through the Dense layer
        return self.dense(inputs)

class Count(tf.keras.metrics.Metric):
    """
    Metric which counts the number of examples seen during training (or evaluation).
    This metric accumulates the batch sizes seen so far and resets on epoch boundary.
    """

    def __init__(self, name='counter', dtype=tf.int64, **kwargs):
        super(Count, self).__init__(name=name, dtype=dtype, **kwargs)
        # Register a state variable 'count' initialized to zero
        self.count = self.add_weight(name='count', initializer='zeros', dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true could be nested, flatten just in case for multi-output models
        first_tensor = tf.nest.flatten(y_true)[0]
        # Get batch size of current update
        batch_size = tf.shape(first_tensor)[0]
        # Accumulate batch size in the count variable cast to metric dtype
        self.count.assign_add(tf.cast(batch_size, dtype=self.dtype))

    def result(self):
        # Return the current count (total samples seen so far)
        return self.count

    def reset_states(self):
        # Reset the count at the end of an epoch
        self.count.assign(0)

def my_model_function():
    """
    Returns a compiled instance of MyModel with optimizer and loss configured
    as in the demonstrated example.
    """
    model = MyModel()
    # Compile with SGD optimizer and MSE loss as per example in the issue
    model.compile(optimizer='sgd', loss='mse')
    return model

def GetInput():
    """
    Returns an input tensor matching the expected input shape for MyModel.
    The example data fed to the model in the issue uses shape (10, 10),
    where batch size 10 and 10 features. We'll generate a random tensor of
    shape (batch_size=10, features=10).
    """
    # Assumption: batch size 10, feature size 10, single input tensor
    return tf.random.uniform((10, 10), dtype=tf.float32)

