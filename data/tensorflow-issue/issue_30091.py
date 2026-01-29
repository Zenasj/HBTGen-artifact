# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Inferred MNIST input shape

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define convolutional layers
        self.conv1 = Conv2D(32, (3, 3), input_shape=(28, 28, 1))
        self.activ = Activation('relu')
        self.conv2 = Conv2D(32, (3, 3))
        self.maxpo = MaxPooling2D(pool_size=(2, 2))

        # Define two separate Conv2D layers instead of reusing conv3 to avoid shape mismatch
        self.conv3 = Conv2D(64, (3, 3))
        self.conv3_1 = Conv2D(64, (3, 3))

        # Flatten and dense layers
        self.flatt = Flatten()
        self.dense = Dense(200)
        self.dense1 = Dense(200)
        self.dens2 = Dense(10)

    def call(self, x):
        # Forward pass matching the original intended architecture
        x = self.conv1(x)
        x = self.activ(x)

        x = self.conv2(x)
        x = self.activ(x)
        x = self.maxpo(x)

        x = self.conv3(x)
        x = self.activ(x)
        x = self.conv3_1(x)  # Use a separate conv layer to avoid shape incompatibilities
        x = self.activ(x)
        x = self.maxpo(x)

        x = self.flatt(x)
        x = self.dense(x)
        x = self.activ(x)

        x = self.dense1(x)
        x = self.activ(x)

        return self.dens2(x)


def my_model_function():
    # Simply return an instance of MyModel
    return MyModel()


def GetInput():
    # MNIST images: batch size can be arbitrary; choose 128 for example consistency with the original code
    # Single-channel 28x28 images normalized to float32
    batch_size = 128
    input_tensor = tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
    return input_tensor

