# tf.random.uniform((1, 112, 112, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the same model architecture as in the issue
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(128, 1, activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        # Forward pass mimics create_model() from the issue
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        out = self.fc2(x)
        return out

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Returns a random tensor matching input shape (batch=1, height=112, width=112, channels=3)
    return tf.random.uniform((1, 112, 112, 3), dtype=tf.float32)

