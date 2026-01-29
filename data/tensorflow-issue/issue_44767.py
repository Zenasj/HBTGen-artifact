# tf.random.uniform((B, 20), dtype=tf.float32) ← inferred input shape based on Dense layer input_shape=(20,)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, weights=None):
        super().__init__()
        # We create a Dense layer with 10 units, and optionally initialize weights at creation.
        # The key insight from the issue: weights passed in kwargs to Layer or Dense do NOT
        # initialize the weights automatically unless layer is built/added to a model and build() is called.
        # So here, we forcibly build the layer with a specific input shape,
        # then set weights if provided.
        self.dense = tf.keras.layers.Dense(10, use_bias=True)
        # Build the layer to initialize weights with expected input shape
        self.dense.build((None, 20))  # Batch dimension is None, input dim = 20
        
        # If weights were provided (a list/tuple of numpy array or tensors),
        # we set them explicitly using set_weights to ensure weights are stored
        if weights is not None:
            # Weights should be a list or tuple of [kernel, bias]
            # Otherwise fallback: maybe no bias provided, or weights incomplete
            try:
                self.dense.set_weights(weights)
            except Exception:
                # If weights format is unexpected, ignore or handle gracefully
                pass

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Example weights to initialize the Dense layer with given shapes:
    # kernel shape (20, 10), bias shape (10,)
    kernel = tf.random.uniform(shape=(20, 10), dtype=tf.float32)
    bias = tf.random.uniform(shape=(10,), dtype=tf.float32)
    weights = [kernel, bias]
    # Return an instance of MyModel initialized with these weights
    return MyModel(weights=weights)

def GetInput():
    # Return a random tensor input with shape expected by the Dense layer: (batch_size, 20)
    # Choosing batch_size=4 arbitrarily
    return tf.random.uniform((4, 20), dtype=tf.float32)

