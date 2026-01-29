# tf.random.uniform((112, 32, 32, 3), dtype=tf.float32) <- Inferred from CIFAR-10 dataset shape and batch_size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using BatchNormalization with momentum and epsilon as per the reported values
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.4639004933194679, epsilon=0.6515653837017596)
        # PReLU activation with alpha initialized to zeros
        self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros')
        # Flatten layer to flatten spatial dimensions to vector
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer outputting the logits for num_classes=10
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # Follow the architecture from the issue report
        x = self.bn(inputs, training=training)
        x = self.prelu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate random input tensor mimicking CIFAR-10 batch:
    # batch_size=112, height=32, width=32, channels=3
    # Using uniform float32 input scaled between 0 and 1, matching original preprocessing
    batch_size = 112
    height = 32
    width = 32
    channels = 3
    input_tensor = tf.random.uniform(
        shape=(batch_size, height, width, channels), 
        minval=0.0, maxval=1.0, dtype=tf.float32
    )
    return input_tensor

