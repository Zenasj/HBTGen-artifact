# tf.random.uniform((batch_size, 28, 28), dtype=tf.float32) ← Inferred input shape for MNIST images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer to transform (28,28) image into a vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Single Dense layer with sigmoid activation (binary classification)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Instantiate the model and compile it with Adam optimizer and binary crossentropy loss.
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model

def GetInput():
    # Return a batch of random input images matching expected input shape (batch_size, 28, 28)
    # Using batch size 32 by default here; dtype float32 as required.
    batch_size = 32
    # Random float images scaled approx same as MNIST raw data (0-255), 
    # but just simulation for testing the model call/compile.
    import numpy as np
    return tf.convert_to_tensor(np.random.uniform(0, 255, size=(batch_size, 28, 28)).astype('float32'))

