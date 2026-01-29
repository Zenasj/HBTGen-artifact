# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the Conv2D + ReLU + MaxPooling layers
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation=None)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.maxpool1 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation=None)
        self.relu2 = tf.keras.layers.Activation('relu')
        # Output layer with softmax for 10 classes (MNIST)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu2(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Following the MNIST input shape: batch size unknown (use 64 for default), 28x28 grayscale images (1 channel)
    # Normally, input pixels are scaled to [0,1], but model works fine either way for inference.
    batch_size = 64  # default batch size per replica in original script
    # Generate random float32 tensor scaled to [0,1]
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

