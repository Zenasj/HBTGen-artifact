# tf.random.uniform((32, 28, 28), dtype=tf.float32) ← inferred input shape from original issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the Sequential model structure in the issue:
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits layer
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        # Output logits directly; loss function expects from_logits=True
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    In line with the issue, this model can be compiled with:
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor compatible with the model.
    The original example used a batch of 32 samples of 28x28 images.
    Using float32 values analogous to normalized image data.
    """
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

