# tf.random.uniform((1, 28, 28, 16), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with dilation_rate=(2,2), 16 filters, kernel size 7x7,
        # bias initialized to ones, input shape (28,28,16)
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=7,
            dilation_rate=(2, 2),
            use_bias=True,
            bias_initializer='ones',
            input_shape=(28, 28, 16)
        )

    def call(self, inputs):
        return self.conv(inputs)

def my_model_function():
    # Return an instance of MyModel; weights are initialized randomly except bias which is ones
    return MyModel()

def GetInput():
    # Return a random tensor compatible with input_shape=(1, 28, 28, 16) of type float32
    return tf.random.uniform(shape=(1, 28, 28, 16), minval=-1.0, maxval=1.0, dtype=tf.float32)

