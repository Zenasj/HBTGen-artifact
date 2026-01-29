# tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # ThresholdedReLU layer with theta=0.3597445834106594 as in original code
        self.thresholded_relu = tf.keras.layers.ThresholdedReLU(theta=0.3597445834106594)
        # MaxPooling2D with pool_size=(1,1) and strides=(1,1) and padding='valid'
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(1,1), strides=(1,1), padding='valid')
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer with softplus activation
        # This is the main cause of unsupported op in TFLite from the issue
        self.dense = tf.keras.layers.Dense(10, activation='softplus')
        
    def call(self, inputs, training=False):
        x = self.thresholded_relu(inputs)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Based on original code: batch_size=122, input shape = (28, 28, 1), dtype float32 normalized between 0-1
    batch_size = 122
    height = 28
    width = 28
    channels = 1
    # Use uniform random inputs as a stand-in for the normalized image input tensor
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

