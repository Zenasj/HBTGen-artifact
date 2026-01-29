# tf.random.normal((B, 10, 2), dtype=tf.float32)

import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Submodule layer that samples from a Normal distribution with shape [batch_size, 2]
        self.normal_layer = self.NormalSampler()

    class NormalSampler(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()

        def build(self, input_shape):
            super().build(input_shape)
            # Create a Normal distribution with means and stddevs matching last dim of input
            # Assuming last dim = 2 in input_shape to align with example, else will broadcast
            self.normal = tfd.Normal(loc=[1.] * input_shape[-1], scale=[1.] * input_shape[-1])

        def call(self, inputs, *args):
            # inputs.shape[0] is typically None because batch dim unknown at graph build time.
            # We use tf.shape(inputs)[0] to dynamically get batch size during execution.
            dynamic_batch_size = tf.shape(inputs)[0]  # dynamic batch size (tensor)
            # Sample shape (batch_size, 2) from the distribution
            # Because Normal expects sample shape, we pass (dynamic_batch_size, 2)
            return self.normal.sample((dynamic_batch_size, 2))

    def call(self, inputs, *args):
        # Log shapes for debugging
        tf.print("tf.shape inputs:", tf.shape(inputs), " inputs.shape:", inputs.shape)
        sample = self.normal_layer(inputs)
        return sample

    @tf.function
    def train_step(self, data):
        # Access dynamic batch size
        batch_size_dynamic = tf.shape(data)[0]
        # Access static batch size (likely None during graph mode)
        batch_size_static = data.shape[0]
        # Create a random tensor using keras.backend.random_normal with dynamic batch size:
        rnd = keras.backend.random_normal(shape=(batch_size_dynamic, 2))
        # Use tf.print for proper tracing in tf.function
        tf.print("random normal from keras.backend shape:", tf.shape(rnd))
        # Demonstrate TensorArray creation with static batch size (may be None)
        ta = tf.TensorArray(dtype=tf.float32, size=100, element_shape=tf.TensorShape((batch_size_static, 1)))
        # Call model to get normal samples, avoiding ValueError by using dynamic batch size internally
        sample = self(data)
        # Return dummy loss dict as per example
        return {"loss": 1.0}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return inputs matching expected shape: (batch_size, 10, 2)
    # Batch size can be arbitrary, using 100 as in example
    return tf.random.normal((100, 10, 2), dtype=tf.float32)

