# tf.random.uniform((B, 224, 224, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use tf.keras.layers.RandomRotation from the latest stable API
        # According to the issue, RandomRotation layers do not work on TPU compiled models
        # because they rely on ImageProjectiveTransform ops not supported by TPU.
        # But for completeness, we include it here since the original model used it.
        # Note: This layer works on CPU/GPU or inside input pipeline (tf.data) but not TPU model compilation.
        self.random_rotation = tf.keras.layers.RandomRotation((-0.1, 0.1))
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(224, 224))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        # The original example applied RandomRotation inside the model.
        # As discussed in the issue, running this on TPU leads to unsupported ops error.
        # A recommended workaround is to do augmentation outside the compiled model,
        # e.g. in tf.data pipeline, but here we reflect the original architecture.
        x = self.random_rotation(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model input specification:
    # Batch size of 10 (arbitrary), 224x224 image, single channel, float32 values in [0,1]
    # This matches the example in the issue report.
    batch_size = 10  # match the example batch size in the issue
    input_shape = (batch_size, 224, 224, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

