# tf.random.uniform((B, H, W, C), dtype=...) ← assuming input shape (28, 28, 1), batch unknown (B)
import tensorflow as tf

class TPUSafeDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(TPUSafeDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs
        # Using inputs.shape for noise_shape can be None in some dims, so safer to use tf.shape
        noise_shape = tf.shape(inputs)
        random_tensor = tf.random.uniform(noise_shape, dtype=inputs.dtype)
        dropout_mask = tf.cast(random_tensor >= self.rate, inputs.dtype)
        # Scale outputs to keep expectation same
        return inputs * dropout_mask / (1.0 - self.rate)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assuming input shape is (28, 28, 1) based on example and common image sizes
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="gelu")
        # Replace built-in Dropout with TPU-safe dropout to avoid MirroredStrategy dropout variable issue
        self.dropout = TPUSafeDropout(0.2)
        self.dense2 = tf.keras.layers.Dense(128)

    def call(self, inputs, training=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random tensor matching input shape (batch, height, width, channels)
    # Batch size chosen as 4 for example; dtype float32
    return tf.random.uniform((4, 28, 28, 1), dtype=tf.float32)

