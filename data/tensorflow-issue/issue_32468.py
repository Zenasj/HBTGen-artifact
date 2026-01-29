# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates the described Keras Sequential model from the issue:
      Input: (28,28)
      Reshape to (28,28,1)
      Conv2D with 4 filters, kernel (2,2), ReLU activation
      MaxPooling2D with pool size (4,4)
      Flatten
      Dense with 10 units and softmax activation

    According to the issue discussion, the quantized TFLite conversion replaces the 
    first Conv2D with a DepthwiseConv2D with depth_multiplier = filters (4), 
    causing issues due to affine_quantization scale mismatch.

    Here we implement the original model as is, no depthwise convolution substitution,
    to demonstrate the intended forward pass compatible with TF 2.20.0 and XLA.

    Assumptions:
    - Input shape is (batch, 28, 28) (grayscale images as in MNIST).
    - Inputs will be reshaped inside the model.
    - Outputs are logits probability vector (10 classes).
    """
    def __init__(self):
        super().__init__()
        # Layers correspond to the original Keras model
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))
        self.conv2d = tf.keras.layers.Conv2D(filters=4, kernel_size=(2, 2), activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(4,4))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.conv2d(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    """
    Generates a random input tensor compatible with MyModel.

    MyModel expects shape (batch, 28, 28), dtype float32.
    This simulates a batch of grayscale 28x28 images.

    Note: We do not add a channel dimension here, 
    as MyModel.reshape layer will add (1) channel dimension.
    """
    batch_size = 1  # can be any batch size; 1 for example/testing
    # Random float input in the usual [0,1) range, standard for image data
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

