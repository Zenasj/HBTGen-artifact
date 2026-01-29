# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28), grayscale images normalized in [0,1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 28x28 images to 784 vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Output Dense layer with 10 units and softmax activation for classification
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of random inputs consistent with the expected input shape (batch_size, 28, 28)
    # Assume batch size is 32 for example
    batch_size = 32
    # Uniform random float tensor in range [0,1], like normalized image input
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

