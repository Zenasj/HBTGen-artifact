# tf.random.uniform((32, 39, 39, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model recreated from the issue's CreateModel() function, converted to subclassing API
        self.conv2d = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            kernel_initializer='glorot_normal',
            name='images',
            input_shape=(39, 39, 3)  # input shape is recorded here for reference
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            10,
            bias_initializer='zeros',
            kernel_initializer='glorot_normal'
        )
        
    def call(self, inputs, training=False):
        x = tf.cast(inputs, tf.float32)  # Ensure inputs are float32 as in generator
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel, weights initialized by default initializers used
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the shape used in the original script:
    # Batch size 32 (from .batch(32) in dataset), image size (39,39,3), dtype float32
    batch_size = 32
    height = 39
    width = 39
    channels = 3
    # Use uniform random floats as placeholder input similar to what the generator yields (zeros in original)
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

