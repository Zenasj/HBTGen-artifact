# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← Input shape inferred from the Flatten input layer in the original Sequential model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Original model structure from issue:
        # Flatten input (28,28)
        # Dense(128, activation=None)
        # ELU activation
        # Dense(10, activation='softmax')

        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation=None)
        self.elu = tf.keras.layers.ELU()
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.elu(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Return a new instance of MyModel.
    # No pretrained weights were mentioned in the issue, so model is returned with default initialization.
    return MyModel()


def GetInput():
    # Return a random float32 input tensor matching the model input (B=1,H=28,W=28)
    # The original example used np.random.randn(1, 28, 28) to generate representative data.
    # Here, use uniform float32 for simplicity and reproducibility.
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

