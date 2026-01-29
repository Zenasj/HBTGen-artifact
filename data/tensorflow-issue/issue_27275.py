# tf.random.uniform((B, 2), dtype=tf.float32)  ← inferred input shape from example x = np.zeros((1, 2))

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()

    def call(self, inputs, training=None):
        # This layer receives the training flag correctly.
        # Using tf.print for TF static graph compatibility and clarity.
        tf.print("layer training arg:", training)
        return inputs

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = MyLayer()

    def call(self, inputs, training=None):
        # The core issue reported was that the "training" argument to call() becomes None
        # when the model is used in model.fit(), but layer submodules get it correctly.
        # This implementation shows the training arg being passed correctly if propagated.
        tf.print("model training arg:", training)
        x = self.l1(inputs, training=training)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # The original example uses input shape (1, 2) with dtype float32
    # We'll generate a random uniform float32 tensor matching that shape.
    # Using batch size = 1 and features = 2, as in original example.
    return tf.random.uniform((1, 2), dtype=tf.float32)

