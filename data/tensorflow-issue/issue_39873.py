# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape from MNIST example (batch size B, height 28, width 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # MNIST-like classifier as given in the issue, using Sequential layers internally for simplicity
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        probs = self.softmax(logits)
        return probs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random uniform float tensor with batch size of 1 and MNIST image shape 28x28
    # The original training normalized MNIST images to [0,1], so uniform is appropriate here
    return tf.random.uniform((1, 28, 28), minval=0.0, maxval=1.0, dtype=tf.float32)

