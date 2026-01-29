# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Four Conv2D layers with 96 filters, kernel_size=3, strides=2, padding='same'
        self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")
        
        # Global Average Pooling 2D
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        # Flatten layer (sometimes redundant after GAP, but preserved here per original code)
        self.flatten = tf.keras.layers.Flatten()
        
        # Final Dense layer with 10 units for CIFAR10 classes
        self.dense = tf.keras.layers.Dense(10)
        # Softmax activation layer for classification probabilities
        self.softmax = tf.keras.layers.Activation(tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Creates a random float32 tensor with shape matching CIFAR10 input: (batch_size, 32, 32, 3)
    # batch_size is arbitrarily chosen as 8 here; can be adjusted as needed
    return tf.random.uniform((8, 32, 32, 3), dtype=tf.float32)

