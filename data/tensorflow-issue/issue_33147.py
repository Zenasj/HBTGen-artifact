# tf.random.uniform((5, 10), dtype=tf.float32) ← Input shape inferred from example: batch_size=5, feature_dim=10

import tensorflow as tf

class A(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(A, self).__init__()
        self.layer = layer

    def call(self, inputs):
        return self.layer(inputs)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define Dense layer with L1 kernel regularizer
        self.obj = tf.keras.layers.Dense(13, kernel_regularizer=tf.keras.regularizers.l1(5))
        # Wrap self.obj inside layer A as a nested layer
        self.layerB = A(self.obj)

    def call(self, inputs):
        # Forward pass through nested layers
        return self.layerB(inputs)

def my_model_function():
    # Initialize and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the model input shape: (5, 10)
    # Using float32 to match tf.ones dtype in original example
    return tf.random.uniform((5, 10), dtype=tf.float32)

