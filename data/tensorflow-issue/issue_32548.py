# tf.random.uniform((64, 28, 28, 1), dtype=tf.float32) ← Input shape assumed from batch size 64, MNIST 28x28 grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the model from issue: 
        # A Conv2D -> MaxPooling2D -> Flatten -> Dense(128 relu) -> Dense(10 softmax)
        self.conv2d = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel as asked
    return MyModel()

def GetInput():
    # Generate a random tensor simulating MNIST batch input, batch size 64, shape (64,28,28,1)
    # Use float32 dtype as expected by Conv2D
    return tf.random.uniform((64, 28, 28, 1), dtype=tf.float32)

