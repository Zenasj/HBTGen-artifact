# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model definition matches the original MNIST CNN from the issue
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu)
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        # Forward pass matching the sequential model from the issue
        x = self.reshape(inputs)
        x = self.conv2d(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return a fresh MyModel instance.
    # No pretrained weights are provided; user would need to train separately.
    return MyModel()

def GetInput():
    # The model expects input shape (batch_size, 28, 28) with float values normalized [0,1].
    # According to the MNIST dataset shape used in the issue, batch input can be (B, 28, 28).
    # Use batch size 1 here as an example.
    input_shape = (1, 28, 28)
    # Generate random float32 input normalized between 0 and 1
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

