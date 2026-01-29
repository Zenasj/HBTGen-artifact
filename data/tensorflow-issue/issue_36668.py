# tf.random.normal((100, 5), dtype=tf.float32) ← input shape is (100, 5), float32 tensor

import tensorflow as tf

class InnerLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(InnerLayer, self).__init__()

    def call(self, inputs):
        # Metric that should always be 1.0
        self.add_metric(tf.constant(1.0), aggregation='mean', name='should_be_1')
        return inputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inner = InnerLayer()

    def call(self, inputs):
        # Metric that should always be 2.0
        self.add_metric(tf.constant(2.0), aggregation='mean', name='should_be_2')
        return self.inner(inputs)

def my_model_function():
    # Return an instance of MyModel; no special weights needed
    return MyModel()

def GetInput():
    # Return random normal input matching model input shape (100, 5)
    return tf.random.normal((100, 5), dtype=tf.float32)

