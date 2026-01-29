# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential MNIST model as described in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random tensor simulating a batch of grayscale MNIST images normalized to [0,1].
    # Batch size is chosen as 64 here to match batch size per replica from the issue.
    batch_size = 64  # picked from BATCH_SIZE_PER_REPLICA
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32, minval=0.0, maxval=1.0)

