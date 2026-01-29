# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← inferred input shape for typical 28x28 image batch (grayscale implied)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent of the Sequential model from the issue:
        # Flatten input from (28,28) to 784
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense 128 with relu activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout (rate = 0.2)
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output dense layer 10 classes with softmax
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected input of shape (batch_size, 28, 28)
    # Assuming default batch size 32 for usability; dtype float32 as typical for TF input
    batch_size = 32
    return tf.random.uniform(shape=(batch_size, 28, 28), dtype=tf.float32)

