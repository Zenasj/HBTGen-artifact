# tf.random.normal((8, 32, 32, 3), dtype=tf.float32) ← Inferred input shape and dtype from provided example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a Conv2D + GlobalAveragePooling + Dense architecture with kernel regularizer
        self.conv = tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        )
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(units=4, activation=None)
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        logits = self.dense(x)
        return logits

def my_model_function():
    """
    Return an instance of MyModel with kernel regularization set so regularization losses are accumulated.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor that is compatible with MyModel's expected input.
    Shape: (batch=8, height=32, width=32, channels=3)
    dtype: float32
    """
    return tf.random.normal((8, 32, 32, 3), dtype=tf.float32)

