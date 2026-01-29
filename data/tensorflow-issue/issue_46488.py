# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original model uses Conv2D -> MaxPooling2D -> BatchNormalization -> Flatten -> Dense -> Dense
        # This model replicates that architecture exactly.
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        # Explicitly pass training to batch normalization
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.bn(x, training=training)  # BN behaves differently in train vs eval
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel for usage.
    return MyModel()

def GetInput():
    # Returns a batch of random float32 tensors shaped (batch_size, 28, 28, 1)
    # Random uniform in [0,1) to simulate grayscale images for MNIST-like images
    batch_size = 32  # assume typical batch size, can be changed as needed
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

