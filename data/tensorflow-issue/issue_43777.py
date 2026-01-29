# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape from MNIST dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent model to the original Sequential with Flatten + Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Create an instance of MyModel, compiled with same optimizer, loss, metrics
    model = MyModel()
    model.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def GetInput():
    # Return a batch of random inputs matching MNIST images shape (batch size 32 assumed)
    # MNIST images are 28x28 grayscale
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

