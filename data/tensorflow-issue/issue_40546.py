# tf.random.uniform((None,), dtype=tf.float32)
import tensorflow as tf
from functools import wraps
from typing import Callable

class SomeClass:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"SomeClass({self.x})"

def main_call(call: Callable) -> Callable:
    """
    Decorator for __call__ that prevents TensorFlow from auto-decorating call with tf.function.
    It manually handles building the model on first call and then runs the original call method.
    This circumvents errors when call returns a non-tensor object.
    """
    @wraps(call)
    def decorated_call(self, *args, **kwargs):
        with tf.name_scope(self.name_scope()):
            if not self.built:
                self.build([])
                self.built = True
            return call(self, *args, **kwargs)
    return decorated_call

class MyModel(tf.keras.Model):
    def build(self, input_shape):
        # Create a Variable 'a' initialized with 1.0
        self.a = tf.Variable(1.0, name="a")

    def serve(self, x):
        # Simple method to add variable 'a' to input
        return x + self.a

    @main_call
    def __call__(self, x):
        """
        call method returns an instance of SomeClass (non-tensor), so it can't be decorated by tf.function.
        This is why we override __call__ with decorator to prevent TensorFlow from tracing it automatically.
        """
        x = tf.cast(x, tf.float32)
        return SomeClass(x + self.a)

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random uniform tensor of shape [None] (dynamic 1D tensor) matching serve input signature
    # Here we assume a batch size of 4 for concrete example.
    return tf.random.uniform((4,), dtype=tf.float32)

