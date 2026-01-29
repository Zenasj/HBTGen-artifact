# tf.random.uniform((1, 512, 512, 3), dtype=tf.float32) ← Input shape for pretrained ResNet101 with imagenet weights, typical HWC format

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load pretrained ResNet101 without top classification layers
        # Use imagenet weights as per original issue description
        self._model = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')
        self._regularizer = tf.keras.regularizers.l2(1e-5)

    def build(self, input_shape=None):
        # Apply L2 kernel regularizer to Conv2D layers of the pretrained model
        # The kernel_regularizer must be set before calling build or before weights are created.
        # However, since the model is pretrained and weights exist, setting kernel_regularizer here
        # will NOT automatically create losses until the weights are "re-built" or the model runs.
        # To trigger regularization losses, call the model on input and then access losses.
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = self._regularizer
        # Call build on the underlying model to finalize shapes if input_shape provided
        if input_shape is not None:
            self._model.build(input_shape)

    def call(self, x, training=False):
        # Calling the wrapped pretrained model will cause regularization losses to be generated based on kernel_regularizer
        return self._model(x, training=training)


def my_model_function():
    # Create an instance of MyModel, build with example input shape, so losses register
    model = MyModel()
    # Build model with expected input shape (batch, height, width, channels)
    # Required to set the shapes and finalize variables before first call
    model.build((None, 512, 512, 3))
    return model

def GetInput():
    # Return random input tensor of shape matching ResNet101 input (batch=1, H=512, W=512, C=3)
    return tf.random.uniform((1, 512, 512, 3), dtype=tf.float32)

