# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← B is batch size (e.g., 1 or more)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the MnistSequential model architecture as in the issue
        self.conv1 = tf.keras.layers.Conv2D(
            32, 5,
            padding='same',
            activation='relu',
            use_bias=False,
            input_shape=(28, 28, 1),
        )
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(
            64, 5,
            padding='same',
            activation='relu',
            use_bias=False,
        )
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights are uninitialized here
    # User can train or load weights as needed
    return MyModel()

def GetInput():
    # Returns a random float32 tensor matching the input shape (1, 28, 28, 1)
    # This shape aligns with the expected input of MyModel.
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

