# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← input shape: (batch_size, 400, 400, 1) assumed

import tensorflow as tf

def _crop_and_concat(inputs, residual_input):
    # To avoid the division by NoneType as in original issue, calculate scale factor dynamically at runtime
    # Use tf.shape to get dynamic shapes instead of static shape which may have None
    inputs_shape = tf.shape(inputs)
    residual_shape = tf.shape(residual_input)
    factor = tf.cast(inputs_shape[1], tf.float32) / tf.cast(residual_shape[1], tf.float32)
    cropped_residual = tf.image.central_crop(residual_input, factor)
    return tf.concat([inputs, cropped_residual], axis=-1)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(3, 3),
                                            activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(3, 3),
                                            activation=tf.nn.relu)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=2)
        self.deconv = tf.keras.layers.Conv2DTranspose(filters=16,
                                                      kernel_size=(2, 2),
                                                      strides=(2, 2),
                                                      padding='same',
                                                      activation=tf.nn.relu)
        self.conv3 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(3, 3),
                                            activation=tf.nn.relu)

    @tf.function  # Enables tracing and saves concrete function shapes correctly
    def call(self, x):
        # Using tf.print instead of print for compatibility inside tf.function if needed
        tf.print(">>> Input Shape", tf.shape(x))
        out = self.conv1(x)
        tf.print(">>> conv1 Shape", tf.shape(out))
        skip = self.conv2(out)
        tf.print(">>> conv2 Shape", tf.shape(skip))
        out = self.maxpool(skip)
        tf.print(">>> maxpool Shape", tf.shape(out))
        out = self.deconv(out)
        tf.print(">>> deconv Shape", tf.shape(out))  # Critical spot where shape was losing spatial dims in saved model
        out = self.conv3(out)
        out = _crop_and_concat(out, skip)

        return out

def my_model_function():
    # Instantiate and return the model object
    return MyModel()

def GetInput():
    # Return a random tensor matching expected input shape for the model
    # Using float32 tensor, batch size = 1, with 400x400 spatial dims and 1 channel (grayscale)
    return tf.random.uniform((1, 400, 400, 1), dtype=tf.float32)

