# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(784, activation="relu")
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))

    @tf.function
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.reshape(x)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching the expected input shape of MyModel.
    # From the issue context, input is MNIST images normalized roughly in [-1,1], shape (B, 28, 28, 1).
    # Batch size is set to 1 to simulate the problematic case.
    BATCH_SIZE = 1
    input_tensor = tf.random.uniform(
        (BATCH_SIZE, 28, 28, 1),
        minval=0,
        maxval=255,
        dtype=tf.float32
    )
    # Normalize input as done in map_function in the issue: (x - 127.5) / 127.5
    input_tensor = (input_tensor - 127.5) / 127.5
    return input_tensor

