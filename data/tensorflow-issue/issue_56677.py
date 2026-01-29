# tf.random.uniform((B, 10), dtype=tf.float32)  ← Input shape inferred from the example inputs and model definition

import tensorflow as tf

def test(values):
    # Dummy function as per the original example; returns zeros_like input
    vector = tf.zeros_like(values)
    return vector


class TestLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestLayer, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        # Logs execution to illustrate call() invocation
        tf.print("=============TestLayer.call()============")
        emb_vector = test(values=inputs)
        return emb_vector


class Demo(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Demo, self).__init__(**kwargs)
        self.test_layer = TestLayer()
        self.dense_layer = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer="ones",
            bias_initializer="zeros",
        )

    def call(self, inputs, training=False):
        # Logs execution to illustrate call() invocation
        tf.print("=============Demo.call()============")
        vector = self.test_layer(inputs)
        logit = self.dense_layer(vector)
        return logit, vector


class MyModel(tf.keras.Model):
    """
    This model fuses the behavior of the original Demo model,
    explicitly showing the connection to the TestLayer and the dense layer.
    It retains the logging and behavior exactly as in the example.

    The forward pass returns a tuple (logit, vector) where:
     - logit is the dense layer output
     - vector is the zero vector produced by test()
    """

    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Instantiate the Demo model as a submodule to clearly separate logic
        self.demo_model = Demo()

    def call(self, inputs, training=False):
        # Forward inputs through the internally encapsulated Demo model
        # The internal Demo model call will produce the logged outputs as in the issue
        return self.demo_model(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a valid random input tensor matching the expected shape (B, 10)
    # The example uses global_batch_size=16384 and input shape (10,)
    # We choose a smaller batch size here for typical usage, can be adjusted as needed.
    batch_size = 32
    input_shape = (batch_size, 10)
    return tf.random.uniform(input_shape, dtype=tf.float32)

