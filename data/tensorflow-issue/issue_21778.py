# tf.random.uniform((32, 8, 8, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reuse of MaxPooling2D layer twice is the core of the issue discussed.
        # To properly reuse layers in subclassed Model, we explicitly call them twice in sequence.
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # Layer reuse: apply the same max_pool layer twice explicitly
        x = self.max_pool(inputs)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel 
    # No weights loading is included here since none were provided.
    return MyModel()

def GetInput():
    # Return a random tensor input matching (batch=32, height=8, width=8, channels=1)
    # This matches the example usage in the issue.
    return tf.random.uniform((32, 8, 8, 1), dtype=tf.float32)

