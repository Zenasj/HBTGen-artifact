# tf.random.normal((64, 512), dtype=tf.float32) ← inferred input shape from examples

import tensorflow as tf

class TestLayer2(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer2, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)

    def call(self, x):
        x = self.dense1(x)
        return x

class TestLayer1(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(TestLayer1, self).__init__()
        # Workaround to register sublayers properly so that trainable_variables and param counting works
        self.layer_names = ['layer_' + str(i) for i in range(2)]
        for layer_name in self.layer_names:
            # Use setattr to register layers so that Keras tracks them
            self.__setattr__(layer_name, TestLayer2(dim))

    def call(self, x):
        for layer_name in self.layer_names:
            x = self.__getattribute__(layer_name)(x)
        return x

class MyModel(tf.keras.Model):
    def __init__(self, dim=512):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.layer = TestLayer1(dim * 2)

    def call(self, x):
        x = self.dense1(x)
        x = self.layer(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default dim=512 as per examples
    return MyModel(512)

def GetInput():
    # Return a random input tensor matching the expected input shape [64, 512]
    # This input shape is taken from the examples where input was tf.random.normal([64, 512])
    return tf.random.normal([64, 512], dtype=tf.float32)

