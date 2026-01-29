# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset shape (28x28 grayscale images)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example model in the issue:
        # Sequential model with Flatten -> Dense(128, relu) -> Dropout(0.2) -> Dense(10, softmax)
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
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # MNIST images are 28x28 grayscale images normalized in [0,1].
    # batch size is set arbitrarily, e.g., 32
    batch_size = 32
    # Inputs should be float32 normalized images like in the example
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

