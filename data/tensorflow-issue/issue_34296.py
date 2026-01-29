# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← input shape inferred from the MNIST example (batch=1, 28x28 grayscale image)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the same simple MNIST classifier model from the issue example
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 input tensor shaped (1, 28, 28)
    # Match the example input for model prediction and tflite conversion
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

