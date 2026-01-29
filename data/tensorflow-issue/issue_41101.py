# tf.random.uniform((B, 28, 28), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the CNN model per the Keras tutorial example for MNIST
        self.reshape_layer = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = inputs
        x = self.reshape_layer(x)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits


def my_model_function():
    """
    Creates and compiles an instance of MyModel following the referenced example:
    - Loss: SparseCategoricalCrossentropy(from_logits=True)
    - Optimizer: SGD with learning_rate=0.001
    - Metrics: accuracy
    """
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


def GetInput():
    """
    Return a random batch input tensor consistent with expected input of MyModel:
    Shape (batch_size, 28, 28), dtype float32 with normalized pixel values [0,1].
    Here batch_size is set to the global batch size used in the multi-worker example:
    per_worker_batch_size=512, num_workers=2 => global_batch_size=1024
    This choice is to match the original multi-worker dataset batching.
    """
    batch_size = 1024
    # Random values in [0,1], shape (batch_size, 28, 28)
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

