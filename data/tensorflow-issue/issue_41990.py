# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from original MNIST model input shape (28x28 grayscale images)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input 28x28 image to 784 vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Final dense layer with 10 units (logits for 10 classes)
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Returns an instance of the defined model
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch_size=1, 28, 28) with dtype float32, values in [0,1)
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

