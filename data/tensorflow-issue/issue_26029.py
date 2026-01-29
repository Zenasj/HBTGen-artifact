# tf.random.uniform((), dtype=tf.int32) ← The example inputs shown are scalar integers, i.e. shape ()

import tensorflow as tf

class A:
    def foo(self, x):
        return x + 1

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Workaround: store super() reference to avoid super() call inside tf.function,
        # which causes "RuntimeError: super(): __class__ cell not found".
        self._super = super()
    
    @tf.function
    def call(self, x):
        # Use the stored super() reference to call the foo method from A.
        # This avoids issue with autograph and super() syntax.
        return self._super.foo(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return an integer scalar tensor input compatible with MyModel.call
    # We use an int32 scalar tensor, consistent with the example usage (e.g. b.bar(5))
    return tf.random.uniform((), minval=0, maxval=10, dtype=tf.int32)

