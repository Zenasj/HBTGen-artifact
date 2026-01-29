# tf.random.uniform((1, 105), dtype=tf.float32)  # Input shape inferred from original PyTorch model and TFLite sample

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequence of Dense/FullyConnected layers as described in the PyTorch FCResidualBlock
        # with input dimension 105, and hidden dimension 512.
        self.dense1 = tf.keras.layers.Dense(512, activation=None)  # linear activation
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(512, activation='relu')
        self.dense4 = tf.keras.layers.Dense(512, activation='relu')
        self.dense5 = tf.keras.layers.Dense(512, activation=None)  # linear output layer

        # Dropout used in PyTorch training not typically used in inference; omitted here.
        # If needed, can be added with training argument in call or as a separate layer.

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """
        Forward pass mimics the residual block pattern given in the referenced PyTorch code:
        x -> dense1
        y = ReLU(dense2(x)), then ReLU(dense3(y)), y is a residual branch
        x + y
        -> ReLU(dense4)
        -> dense5

        Dropout layers are omitted since this is for inference and TFLite conversion.
        """
        x = self.dense1(inputs)        # linear
        y = self.dense2(x)             # ReLU
        y = self.dense3(y)             # ReLU
        x = x + y                     # residual addition
        x = self.dense4(x)             # ReLU
        out = self.dense5(x)           # linear output
        return out


def my_model_function():
    """
    Returns an instance of MyModel initialized with default random weights.
    Typically, you would load trained weights here if available.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the input shape expected by MyModel.
    As inferred, the input shape is (batch_size=1, input_dim=105) and dtype float32.
    """
    return tf.random.uniform((1, 105), dtype=tf.float32)

