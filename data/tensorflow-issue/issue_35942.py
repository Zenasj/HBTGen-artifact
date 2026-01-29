# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape from example: (28, 28) grayscale images, batch size B unspecified

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model defined corresponds to example in the issue:
        # Flatten(input_shape=(28,28)) + Dense(10, activation softmax)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass
        x = self.flatten(inputs)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile the model to match example: Adam optimizer + sparse_categorical_crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy'
    )
    return model

def GetInput():
    # Return a random tensor with shape (batch_size, 28, 28)
    # Use batch size 32 as a reasonable default
    batch_size = 32
    # Values normalized to [0,1] like the MNIST data preprocessing described
    return tf.random.uniform(shape=(batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

