# tf.random.normal((2, 3), dtype=tf.float32) ← inferred from example usage in the issue's example code

import tensorflow as tf
import threading

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(2)

        # This _thread_local attribute causes pickling issues due to threading.local()
        # It is created by tf.keras.Model internally for tracking but is not serializable by pickle.
        # We keep it here to match the behavior and preserve model functionality.
        self._thread_local = threading.local()

    def call(self, x):
        return self.dense(x)


def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input shape of MyModel
    # In the example, input was shape (2, 3)
    return tf.random.normal((2, 3))

