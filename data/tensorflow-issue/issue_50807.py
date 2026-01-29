# tf.random.uniform((), dtype=tf.int32) ← Input is a scalar integer parameter to the method

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # An integer attribute, similar to the one in MyObject.some_attribute
        self.some_attribute = tf.constant(2)

    @tf.function
    def call(self, param):
        # param is expected as a scalar tensor (tf.int32)
        # Returns the sum of some_attribute + param, mimicking the original example behavior
        return self.some_attribute + param

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a scalar tensor to match the expected input to the model
    # We use tf.constant(3, dtype=tf.int32) to mimic the example param=3
    return tf.constant(3, dtype=tf.int32)

