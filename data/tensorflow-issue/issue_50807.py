import tensorflow as tf


class MyObject(object):
    def __init__(self):
        self.some_attribute = 2

    @tf.function
    def some_tf_function(self, param):
        return self.some_attribute + param


# This works:
obj = MyObject()
obj.some_tf_function(3)  # returns 5

# This throws AttributeError: 'NoneType' has no attribute 'some_attribute'
result = MyObject().some_tf_function(3)