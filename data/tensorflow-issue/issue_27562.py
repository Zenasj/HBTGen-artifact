# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the CNN model architecture as per the example
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel as defined above
    return MyModel()

def GetInput():
    # Return a random input tensor simulating a batch of grayscale 28x28 images
    BATCH_SIZE = 64  # consistent with the issue example batch size
    # The model expects float32 inputs scaled between 0 and 1 - simulate this by uniform distribution
    return tf.random.uniform((BATCH_SIZE, 28, 28, 1), minval=0.0, maxval=1.0, dtype=tf.float32)

