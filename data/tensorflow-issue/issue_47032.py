# tf.random.uniform((2, 10, 10, 3), dtype=tf.uint8) ← Batch of 2 images, shape from dataset batch size and map_fun input

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten and Dense to match the original example model architecture
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        return self.dense(x)


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile model similar to original example with categorical_crossentropy loss and Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


def GetInput():
    # Generate a batch of 2 dummy images of shape (10, 10, 3), uint8 like the py_function output
    # This matches the input shape and dtype expected by MyModel
    batch_size = 2
    height, width, channels = 10, 10, 3
    # Use tf.random.uniform with dtype uint8 (0-255) to simulate dummy images as in original map_fun
    dummy_images = tf.random.uniform(
        shape=(batch_size, height, width, channels),
        minval=0,
        maxval=256,
        dtype=tf.uint8
    )
    return dummy_images

