# tf.random.uniform((1, 227, 227, 3), dtype=tf.float32)
import tensorflow as tf


def conv(x, filter_height, filter_width, num_filters,
         stride_y, stride_x, name, padding='SAME', groups=1):
    """Create a convolution layer similar to the original AlexNet conv layer.
    Groups split the inputs and weights for grouped convolution.
    """
    input_channels = int(x.shape[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.name_scope(name):
        weights_shape = [filter_height, filter_width,
                         input_channels // groups, num_filters]
        weights = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=0.05),
                              name='weights')
        biases = tf.Variable(tf.zeros([num_filters]), name='biases')

        if groups == 1:
            conv_out = convolve(x, weights)
        else:
            # Split input and weights and convolve separately for groups
            input_groups = tf.split(x, num_or_size_splits=groups, axis=3)
            weight_groups = tf.split(weights, num_or_size_splits=groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv_out = tf.concat(output_groups, axis=3)

        biased = tf.nn.bias_add(conv_out, biases)
        relu = tf.nn.relu(biased, name=name + '_relu')

    return relu


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    return tf.nn.max_pool2d(x, ksize=[filter_height, filter_width],
                            strides=[stride_y, stride_x], padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    # Local Response Normalization
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


class MyModel(tf.keras.Model):
    """
    A Keras reimplementation of the core AlexNet feature extractor described in the issue,
    matching the fixed input shape of [1, 227, 227, 3] and producing flattened output of size 9216.
    This version fuses the architecture and builds the layers within a tf.keras.Model subclass,
    dropping dropout and fully connected layers to align with the code in the issue.

    The dynamic number of groups for convolution layers is implemented as in original AlexNet.

    Output is the flattened tensor after the last max pool layer.
    """

    def __init__(self):
        super().__init__()

        # Layer configuration according to the original script:
        # conv1: 96 filters, 11x11, stride 4, padding VALID
        # lrn + maxpool 3x3 stride 2 valid
        self.conv1 = lambda x: conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        self.norm1 = lambda x: lrn(x, 2, 2e-05, 0.75, name='norm1')
        self.pool1 = lambda x: max_pool(x, 3, 3, 2, 2, padding='VALID', name='pool1')

        # conv2: 256 filters, 5x5, stride 1, groups=2
        # lrn + maxpool 3x3 stride 2 valid
        self.conv2 = lambda x: conv(x, 5, 5, 256, 1, 1, groups=2, name='conv2')
        self.norm2 = lambda x: lrn(x, 2, 2e-05, 0.75, name='norm2')
        self.pool2 = lambda x: max_pool(x, 3, 3, 2, 2, padding='VALID', name='pool2')

        # conv3: 384 filters, 3x3, stride 1
        self.conv3 = lambda x: conv(x, 3, 3, 384, 1, 1, name='conv3')

        # conv4: 384 filters, 3x3, stride 1, groups=2
        self.conv4 = lambda x: conv(x, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # conv5: 256 filters, 3x3, stride 1, groups=2
        self.conv5 = lambda x: conv(x, 3, 3, 256, 1, 1, groups=2, name='conv5')

        # maxpool 3x3 stride 2 valid
        self.pool5 = lambda x: max_pool(x, 3, 3, 2, 2, padding='VALID', name='pool5')

    def call(self, inputs):
        # inputs expected shape: (1, 227, 227, 3)
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        # Flatten to shape (1, 9216) as per original model
        x = tf.reshape(x, [1, 9216], name='output')
        return x


def my_model_function():
    """
    Returns an instance of the AlexNet-like MyModel.
    This model uses randomly initialized weights (as no weights loading was fully reproducible here).
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the input shape expected by MyModel:
    Batch size =1, Height=227, Width=227, Channels=3, dtype float32.
    """
    # Random uniform input between 0 and 255, matching typical RGB image normalized loosely.
    return tf.random.uniform((1, 227, 227, 3), minval=0, maxval=255, dtype=tf.float32)

