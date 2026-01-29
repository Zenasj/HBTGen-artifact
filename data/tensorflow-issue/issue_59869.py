# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28) grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple MNIST classifier model replicating the structure from the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Create and compile an instance of MyModel
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def GetInput():
    # Return a random input tensor matching MNIST grayscale images (batch size arbitrarily 32)
    # Values normalized between 0 and 1 as in the example preprocessing
    return tf.random.uniform(shape=(32, 28, 28), minval=0, maxval=1, dtype=tf.float32)

