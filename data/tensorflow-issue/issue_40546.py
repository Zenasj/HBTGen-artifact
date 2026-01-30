from tensorflow import keras

import tensorflow as tf


class SomeClass:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"SomeClass({self.x})"


class MyModel(tf.keras.Model):
    def build(self, input_shape):
        self.a = tf.Variable(1.0, name="a")

    def serve(self, x):
        return x + self.a

    def call(self, x):
        """This method can't be decorated by `tf.function` because it returns an object that is not a Tensor.
        It should not matter, because we do not want to save it.
        """
        x = tf.cast(x, tf.float32)
        return SomeClass(x + self.a)


model = MyModel()
print("Call the model in order to build it: ", model(4))

signatures = {"my_stuff": tf.function(model.serve, input_signature=[tf.TensorSpec([None], tf.float32)])}
tf.saved_model.save(model, export_dir="", signatures=signatures)

from functools import wraps
from typing import Callable

import tensorflow as tf


class SomeClass:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return f"SomeClass({self.x})"


def main_call(call: Callable) -> Callable:
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
        self.a = tf.Variable(1.0, name="a")

    def serve(self, x):
        return x + self.a

    @main_call
    def __call__(self, x):
        """This method can't be decorated by `tf.function` because it returns an object that is not a Tensor.
        It should not matter, because we do not want to save it.
        """
        x = tf.cast(x, tf.float32)
        return SomeClass(x + self.a)


model = MyModel()
print("Call the model in order to build it: ", model(4))

signatures = {"my_stuff": tf.function(model.serve, input_signature=[tf.TensorSpec([None], tf.float32)])}
model.built = False
tf.saved_model.save(model, export_dir="d:/tmp_model", signatures=signatures)