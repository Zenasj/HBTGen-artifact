# tf.random.uniform((8, 644, 644, 1), dtype=tf.float32) ← inferred input shape from logs and placeholders

import tensorflow as tf
from tensorflow.keras.layers import Conv2D as KerasConv2D

# We define a Conv2D helper to mimic the original style (with name scope) for conv layers
def Conv2D(name, x, filters, kernel_size, strides):
    # Using Keras Conv2D layer with given params and 'same' padding for simplicity
    layer = KerasConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                        padding='same', activation='relu', name=name)
    return layer(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Re-create convolutional layers to match reported naming and parameters
        self.conv1 = KerasConv2D(filters=16, kernel_size=3, strides=(2,2), padding='same', activation='relu', name='conv1')
        self.conv2 = KerasConv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu', name='conv2')
        self.conv3 = KerasConv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name='conv3')
        self.conv4 = KerasConv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu', name='conv4')
        # For demonstration, we create a dummy branch output that might be concatenated later
        # Here we simulate multiple branches as different conv layers and concatenate
        self.branch1 = KerasConv2D(filters=2, kernel_size=1, strides=1, padding='same', activation='linear', name='branch1')
        self.branch2 = KerasConv2D(filters=2, kernel_size=1, strides=1, padding='same', activation='linear', name='branch2')
        self.branch3 = KerasConv2D(filters=2, kernel_size=1, strides=1, padding='same', activation='linear', name='branch3')

    def call(self, inputs, training=False):
        """
        Forward pass:
        - inputs: Tensor shaped [batch, height, width, channels] expected as (8, 644, 644, 1)
        - implements conv layers + dummy concat branch output
        - computes gradients of a scalar prediction w.r.t input images
        """
        images = inputs  # Expect single input tensor with shape [8, 644, 644, 1]

        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Generate dummy multi-branch predictions simulating class prediction with background
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # Concatenate along the channel dimension (axis=-1)
        class_prediction_with_background = tf.concat([b1, b2, b3], axis=-1)  # shape [batch, H, W, 6]

        # Suppose channel 0 is background, channel 1 is foreground for each branch
        # Extract only the foreground scores from each branch for a face detector
        # The original snippet: class_prediction_with_background[:,:,1]
        # Here, we take the 1st channel of each 2-channel branch:
        # channels arranged as b1(0:2), b2(2:4), b3(4:6)
        fg_channels = class_prediction_with_background[..., 1::2]  # shape [batch, H, W, 3]

        # For simplicity, reduce to scalar by averaging all foreground scores
        pred_scalar = tf.reduce_mean(fg_channels, axis=[1,2,3])  # shape [batch]

        # To compute gradient of sum over batch scalar w.r.t. input images
        total_pred = tf.reduce_sum(pred_scalar)  # scalar

        # tf.gradients returns a list of tensors, get first element for images gradient
        gradients = tf.gradients(total_pred, images)[0]

        # Regularization term is mean of gradients (scalar) - this mimics the original regular_vals
        regular_vals = tf.reduce_mean(gradients)

        # For demonstration, output a dict including predictions and regularization term
        # In a real training scenario, loss would include regular_vals added to classification loss
        return {
            'class_prediction_with_background': class_prediction_with_background,
            'foreground_scores': fg_channels,
            'regularization_term': regular_vals,
            'input_gradients': gradients
        }

def my_model_function():
    # Return a compiled instance of MyModel
    model = MyModel()
    return model

def GetInput():
    # Return a random input tensor simulating batch of 8 grayscale images of size 644x644
    # dtype is float32 matching placeholder in original code
    return tf.random.uniform((8, 644, 644, 1), dtype=tf.float32)

