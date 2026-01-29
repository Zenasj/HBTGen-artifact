# tf.random.uniform((1, 28, 28), dtype=tf.float32) ← input shape as (28, 28) grayscale image batch size 1, per the model input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following model from the issue:
        # Sequential model has Input(shape=(28,28)) → Reshape (28,28,1)
        # → Conv2D(12 filters, 3x3, ReLU) → MaxPooling2D(2x2)
        # → Flatten → Dense(10, softmax)
        self.reshape = tf.keras.layers.Reshape(target_shape=(28, 28, 1))
        self.conv = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, 28, 28)
        x = self.reshape(inputs)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching input shape (1, 28, 28) float32 normalized like in the examples
    # Batch size 1 to be compatible with TFLite calibration which expects fixed batch size
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

