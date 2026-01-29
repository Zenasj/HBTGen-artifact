# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST images scaled and batch size context

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from build_and_compile_cnn_model()
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model instance.
    # Note: Compilation with loss/optimizer is done externally usually, but not requested here.
    return MyModel()

def GetInput():
    # Return a random tensor matching expected MNIST input shape: batch size 4 is used as a default example
    BATCH_SIZE = 4
    return tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)

