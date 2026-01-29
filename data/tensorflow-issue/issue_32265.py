# tf.random.uniform((128, 256, 256, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Minimal model replicating the original example: a single Conv2D (1x1) with sigmoid activation
        self.conv = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, inputs, training=False):
        # Forward pass: Apply conv layer
        return self.conv(inputs)


def my_model_function():
    # Instantiate and compile model analogous to original example
    model = MyModel()
    # Compile with adam optimizer, binary crossentropy loss, and accuracy metric
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def GetInput():
    # Return a batch of random input images with shape (128, 256, 256, 3)
    # matching the batch_size and input image shape from the original example
    return tf.random.uniform((128, 256, 256, 3), dtype=tf.float32)

