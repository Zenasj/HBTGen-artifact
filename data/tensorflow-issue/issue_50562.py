# tf.random.uniform((12800, 120, 120, 3), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the example CNN model described in the issue:
        # Conv2D with 256 filters, kernel size 5, stride 2, input shape (120,120,3), activation relu
        # Two Conv2D layers with same spec in sequential
        # Global average pooling
        # Dense output with sigmoid activation for binary classification

        self.conv1 = tf.keras.layers.Conv2D(
            256, kernel_size=5, strides=2, activation='relu', input_shape=(120, 120, 3))
        self.conv2 = tf.keras.layers.Conv2D(
            256, kernel_size=5, strides=2, activation='relu')
        self.gap = tf.keras.layers.GlobalAvgPool2D()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.gap(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape (12800,120,120,3)
    # TF model expects float32 inputs. We create a random tensor of the same shape and dtype.
    return tf.random.uniform((12800, 120, 120, 3), minval=0, maxval=1, dtype=tf.float32)

