# tf.random.uniform((B, 10), dtype=tf.float64) ← inferred input shape from Dense layer input_shape=(10,), dtype float64
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dense layer with float64 dtype and no bias
        self.dense = tf.keras.layers.Dense(
            10,
            use_bias=False,
            dtype=tf.float64,
            input_shape=(10,)
        )
        # Custom GaussianNoise layer fixed for float64 compatibility
        self.noise = GaussianNoiseFloat64(0.0003)

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.noise(x, training=training)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float64 tensor input compatible with MyModel
    # Batch size is arbitrary, here set to 4 for example
    batch_size = 4
    return tf.random.uniform(shape=(batch_size, 10), dtype=tf.float64)

# GaussianNoise with fix: noise generated on each call uses inputs' dtype
class GaussianNoiseFloat64(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            # Generate noise with the same dtype as inputs
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0., stddev=self.stddev, dtype=inputs.dtype)
            return inputs + noise

        return tf.keras.backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config

