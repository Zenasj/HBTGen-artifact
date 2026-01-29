# tf.random.uniform((batch_size, 252, 400, 3), dtype=tf.float32)
import math
import tensorflow as tf

# Assumptions:
# - IMG_HEIGHT = 252, IMG_WIDTH = 400, IMG_CHANNEL = 3 as seen in the comments and reshape shapes.
# - batch_size = 32 (from Input.py batch size) but keep it as a variable for inputs.
# - Based on original code conv2d_transpose and conv2d were used with 2x2 kernels and stride 2 mainly.
# - Linear layer projects input tensor to f_dim*8*s_h16*s_w16 where f_dim=64.
# - Following the final comment, we add explicit tf.reshape after conv2d_transpose and conv2d to ensure shapes are fully defined;
#   this was the solution to 'Conv2DSlowBackpropInput' batch_size mismatch error.
# - We implement the inference model as MyModel(tf.keras.Model).
# - The model inputs a 2D tensor: (batch_size, feature_dim=10), representing "pressure data" features.
# - Outputs a tensor of shape (batch_size, 252, 400, 3).
# - We implement BatchNorm as a tf.keras layer wrapper around tf.keras.layers.BatchNormalization (using tf.contrib is legacy).
# - We replace deprecated APIs (tf.get_variable, tf.variable_scope) with tf.keras layers and variable management.
# - Provide variable summaries as no-ops since tf.summary is part of graph training infrastructure (can be omitted or replaced with tf.print).
# - We fuse all parts into a single cohesive tf.keras.Model class.


# Constants
IMG_HEIGHT = 252
IMG_WIDTH = 400
IMG_CHANNEL = 3
NUM_LABELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL

batch_size = 32  # Default batch size (must match GetInput)
f_dim = 64
channel_dim = IMG_CHANNEL

CONV1_DEPTH = 16
CONV2_DEPTH = 32
CONV3_DEPTH = 64
CONV4_DEPTH = 128
FC_NODE = 512

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# Calculate spatial sizes at different resolutions
s_h, s_w = IMG_HEIGHT, IMG_WIDTH
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9, name=None):
        super().__init__(name=name)
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=momentum, epsilon=epsilon, scale=True)

    def call(self, x, training=True):
        return self.bn(x, training=training)


def variable_summaries(var, name):
    # Placeholder for summary - during training these could be tf.summary.histogram/scalar calls
    # Here, just a no-op or tf.print can be used if desired.
    pass


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define all layers and variables here

        # BatchNorm instances
        self.d_bn_0 = BatchNorm(name="d_bn_0")
        self.d_bn_1 = BatchNorm(name="d_bn_1")
        self.d_bn_2 = BatchNorm(name="d_bn_2")
        self.d_bn_3 = BatchNorm(name="d_bn_3")
        self.d_bn_4 = BatchNorm(name="d_bn_4")
        self.c_bn_0 = BatchNorm(name="c_bn_0")

        # Layers for conv and deconv operations will be implemented as tf.keras.layers.Conv2D and Conv2DTranspose with fixed params

        # Linear projection layer replacing TF linear function
        self.linear_dense = tf.keras.layers.Dense(f_dim * 8 * s_h16 * s_w16, activation=None,
                                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                                  bias_initializer=tf.keras.initializers.Constant(0.1),
                                                  name="linear")

        # Deconvolution (Conv2DTranspose) layers for upsampling stage
        self.deconv1 = tf.keras.layers.Conv2DTranspose(
            filters=f_dim * 4, kernel_size=2, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name="deconv1")

        self.deconv2 = tf.keras.layers.Conv2DTranspose(
            filters=f_dim * 2, kernel_size=2, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name="deconv2")

        self.deconv3 = tf.keras.layers.Conv2DTranspose(
            filters=f_dim, kernel_size=2, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name="deconv3")

        self.deconv4 = tf.keras.layers.Conv2DTranspose(
            filters=channel_dim, kernel_size=2, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name="deconv4")

        # Convolution layers after upsampling
        self.conv1 = tf.keras.layers.Conv2D(
            filters=CONV1_DEPTH, kernel_size=2, strides=2, padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="conv1")

        self.conv2 = tf.keras.layers.Conv2D(
            filters=CONV2_DEPTH, kernel_size=2, strides=2, padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="conv2")

        self.conv3 = tf.keras.layers.Conv2D(
            filters=CONV3_DEPTH, kernel_size=2, strides=2, padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="conv3")

        self.conv4 = tf.keras.layers.Conv2D(
            filters=CONV4_DEPTH, kernel_size=2, strides=2, padding='same',
            activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.0),
            name="conv4")

        # Fully connected layers
        self.fc1 = tf.keras.layers.Dense(
            FC_NODE, activation='relu',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name='fully1')

        self.fc2 = tf.keras.layers.Dense(
            NUM_LABELS, activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.1),
            name='fully2')

    def call(self, input_tensor, training=True):
        """
        input_tensor: shape (batch_size, 10) - 10 feature pressure data per batch item
        returns: tensor with shape (batch_size, 252, 400, 3)
        """
        batch_size_actual = tf.shape(input_tensor)[0]

        # 1) Linear and reshape to 4D tensor
        z = self.linear_dense(input_tensor)  # [batch_size, f_dim*8*s_h16*s_w16]
        h0 = tf.reshape(z, [batch_size_actual, s_h16, s_w16, f_dim * 8])
        h0 = tf.nn.relu(self.d_bn_0(h0, training=training))

        # 2) Upsampling with conv2d_transpose (deconv) layers with reshapes after each as per solution

        # deconv1
        h1 = self.deconv1(h0)
        h1 = tf.reshape(h1, [batch_size_actual, s_h8, s_w8, f_dim * 4])
        h1 = self.d_bn_1(h1, training=training)

        # deconv2
        h2 = self.deconv2(h1)
        h2 = tf.reshape(h2, [batch_size_actual, s_h4, s_w4, f_dim * 2])
        h2 = self.d_bn_2(h2, training=training)

        # deconv3
        h3 = self.deconv3(h2)
        h3 = tf.reshape(h3, [batch_size_actual, s_h2, s_w2, f_dim])
        h3 = self.d_bn_3(h3, training=training)

        # deconv4
        h4 = self.deconv4(h3)
        h4 = tf.reshape(h4, [batch_size_actual, s_h, s_w, channel_dim])
        h4 = self.d_bn_4(h4, training=training)

        # 3) Convolution + max pool layers
        conv1 = self.conv1(h4)
        conv1 = self.c_bn_0(conv1, training=training)
        pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')

        conv2 = self.conv2(pool1)
        pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')

        conv3 = self.conv3(pool2)
        pool3 = tf.nn.max_pool2d(conv3, ksize=2, strides=2, padding='SAME')

        conv4 = self.conv4(pool3)

        # 4) Flatten and Fully connected layers
        pool_shape = tf.shape(conv4)
        batch_size_flat = pool_shape[0]
        nodes = conv4.shape[1] * conv4.shape[2] * conv4.shape[3]  # Static dims for FC layer init
        reshaped = tf.reshape(conv4, [batch_size_flat, nodes])
        fc1 = self.fc1(reshaped)
        fc2 = self.fc2(fc1)

        # 5) Reshape to output image size
        output_array = tf.reshape(fc2, [batch_size_flat, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        return output_array


def my_model_function():
    # Instantiate MyModel
    return MyModel()


def GetInput():
    # Return random input tensor of shape (batch_size, 10) for pressure data features
    # Here batch_size is set to 32 matching original code batch size in input pipeline
    batch_size_val = 32
    feature_dim = 10
    return tf.random.uniform((batch_size_val, feature_dim), dtype=tf.float32)

