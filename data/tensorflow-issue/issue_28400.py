# tf.random.uniform((B, 2560, 8), dtype=tf.float32) ← Using input shape from the reported issue input_shape=(2560,8)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Model replicating the reported Keras Conv1D structure to demonstrate the issue
    with shape inference loss when dilation_rate > 1 in Conv1D with padding='causal'.
    
    This MyModel allows choosing dilation_rate to illustrate the shape difference.
    Forward returns output tensor after softmax.
    """
    def __init__(self, dilation_rate=1):
        super().__init__()
        # Using kernel_regularizer as per original example
        self.conv1d = tf.keras.layers.Conv1D(
            filters=24,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding='causal',
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            input_shape=(2560, 8)
        )
        self.relu = tf.keras.layers.ReLU()
        self.dense = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.relu(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function(dilation_rate=1):
    """
    Returns an instance of MyModel with specified dilation_rate.
    Defaults to dilation_rate=1 (which preserves shape inference as expected).
    """
    return MyModel(dilation_rate=dilation_rate)

def GetInput():
    """
    Returns a random tensor input matching the expected input shape (batch, 2560, 8).
    Batch size is set to 1 to keep it simple; dtype is float32.
    """
    batch_size = 1
    time_steps = 2560
    channels = 8
    return tf.random.uniform((batch_size, time_steps, channels), dtype=tf.float32)

