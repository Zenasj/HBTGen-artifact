# tf.random.uniform((20, 20, 1), dtype=tf.float32) ← inferred input shape from example: (batch=20?, length=20, channels=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue's example model architecture with Conv1D and Conv1DTranspose
        # Note: dilation_rate > 1 for Conv1DTranspose leads to runtime error in current TF versions
        # We keep dilation_rate=2 to reflect the original example and document the limitation
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=3,
            strides=1,
            padding='same',
            dilation_rate=2,
            activation='relu'
        )
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='relu')
        self.dense4 = tf.keras.layers.Dense(20, activation='relu')
        self.reshape = tf.keras.layers.Reshape((20, 1))
        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            dilation_rate=2,  # This triggers error on current TF versions (>1 not supported)
            padding='same',
            activation='relu',
            output_padding=0
        )
        self.flatten2 = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.reshape(x)
        x = self.conv1d_transpose(x)
        x = self.flatten2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, weights uninitialized (random initialization standard)
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected input shape (batch, length, channels)
    # Based on original example: input shape (20, 20, 1)
    # For batch size, here 20 to match example train_data shape (20,20)
    return tf.random.uniform(shape=(20, 20, 1), dtype=tf.float32)

