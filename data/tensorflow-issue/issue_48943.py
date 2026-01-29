# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape from the example code where inputs = tf.keras.Input(10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer from the provided snippet
        self.dense = tf.keras.layers.Dense(10)
        # Lambda layer wrapping my_summary function
        self.my_summary = tf.keras.layers.Lambda(self._my_summary)
        # Custom subclass layer MyLayer
        self.my_layer = MyLayer()

    def _my_summary(self, x):
        # Add scalar summary of mean of x during forward pass
        tf.summary.scalar("mean_functional", tf.reduce_mean(x))
        return x

    def call(self, inputs, training=None):
        x = self.dense(inputs)
        x = self.my_summary(x)
        x = self.my_layer(x)
        return x

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs):
        # Add scalar summary of mean of inputs during forward pass
        tf.summary.scalar("mean_subclass", tf.reduce_mean(inputs))
        return inputs

def my_model_function():
    # Return an instance of MyModel with all layers initialized
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected shape (batch_size=16, features=10), dtype float32
    # Batch size inferred from example dataset batch
    return tf.random.uniform((16, 10), dtype=tf.float32)

