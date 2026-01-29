# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred from example Input layer shape=(2,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, name='MyModel', **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        # Define two Dense layers matching the example
        self.dense1 = tf.keras.layers.Dense(4, name='layer1')
        self.dense2 = tf.keras.layers.Dense(2, name='layer2')

    # Override __call__ instead of call to avoid the inbound nodes error
    def __call__(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape (batch_size, 2)
    # batch size chosen arbitrarily as 3 for demonstration purposes
    batch_size = 3
    return tf.random.uniform((batch_size, 2), dtype=tf.float32)

