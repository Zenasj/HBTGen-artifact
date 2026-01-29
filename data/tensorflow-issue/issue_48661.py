# tf.random.uniform((B, 5, 20), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential model similar to the shared example with Conv1D/MaxPooling/Flatten/Dense
        self.conv1 = tf.keras.layers.Conv1D(32, kernel_size=1, strides=1, activation="relu", padding="same", input_shape=(5, 20))
        self.pool1 = tf.keras.layers.MaxPooling1D(pool_size=3, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(64, kernel_size=1, strides=1, activation="relu", padding="same")
        self.pool2 = tf.keras.layers.MaxPooling1D(pool_size=3, padding="same")
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        # Output layer with 1 unit and softmax activation (as per the reported example, though usually sigmoid is used for binary)
        self.dense2 = tf.keras.layers.Dense(1, activation="softmax")
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size=32, 5, 20)
    # Use float32 dtype to match typical model input
    return tf.random.uniform((32, 5, 20), dtype=tf.float32)

