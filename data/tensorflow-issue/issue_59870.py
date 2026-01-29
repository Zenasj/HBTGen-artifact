# tf.random.uniform((1, 3, 32, 32), dtype=tf.float32) ← Inferred input shape based on PyTorch example (N,C,H,W) rearranged to NHWC for TensorFlow

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Conv2D layer mimicking the first conv from example: in_channels=3, out_channels=16, kernel_size=3, padding=1
        # Padding=1 means 'same' padding in TF. Stride=1 dilation=1 here.
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding='same', dilation_rate=1, activation='relu'
        )
        # The PyTorch example used batch norm and ReLU layers named batch_to_space and space_to_batch,
        # presumably as placeholders or exemplars. Here we just replicate their effect with BatchNormalization and ReLU.
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu_after_bn = tf.keras.layers.Activation('relu')

        # Second conv layer similar: in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=1, padding='same', dilation_rate=1, activation='relu'
        )
        # Flatten before dense layers
        self.flatten = tf.keras.layers.Flatten()

        # The input spatial size in example is 32x32 after conv2 (assuming same padding)
        # So flatten size is 32*32*32 channels
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)  # Output for 10 classes

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.batch_norm(x, training=training)
        x = self.relu_after_bn(x)

        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Model weights not initialized here; user can train or load weights as needed.
    return model


def GetInput():
    # Return a random input tensor matching expected input shape: Batch=1, Height=32, Width=32, Channels=3
    # Using float32 as typical for TF models before quantization
    return tf.random.uniform(shape=(1, 32, 32, 3), dtype=tf.float32)

