# tf.random.uniform((1, 9), dtype=tf.float32) ← The original input is a flat vector of 9 floats, reshaped in model to (1, 3, 3, 1)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Conv2D layer with kernel size 4x4, padding='valid', 1 filter, ReLU activation
        self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=4, padding='valid', activation='relu')

    @tf.function(jit_compile=True)
    def call(self, x):
        # Expect input 'x' to be shape [1, 9], reshape to [1, 3, 3, 1] to match 2D conv input
        x = tf.reshape(x, [1, 3, 3, 1])
        x = self.conv(x)
        return x

def my_model_function():
    # Returns an instance of the model, freshly initialized weights
    return MyModel()

def GetInput():
    # Generate a random tensor of shape [1, 9] to be reshaped inside the model to [1, 3, 3, 1]
    # Using float32 as per original code
    return tf.random.uniform([1, 9], dtype=tf.float32)

