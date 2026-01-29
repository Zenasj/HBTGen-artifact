# tf.random.normal((B=1, H=140, W=140, D=40, C=1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A 3D dilated convolution model that incorporates Weight Standardization
    on convolutional kernels and supports dilation rates > 1 with valid padding 
    to avoid the "No algorithm without scratch worked" error reported on some GPUs.

    The model follows the approximate logic from the original TensorFlow 1.x code:
      - One dilated convolution layer with batch norm and ReLU
      - One final 1x1x1 convolution producing logits for 2 classes
      - Softmax on logits

    Because dilation with padding='SAME' led to errors, this implementation uses
    padding='VALID' and mimics weight standardization.

    Assumptions:
      - Input shape: [batch, height, width, depth, channels] = [1,140,140,40,1] (as per test code)
      - Filters in dilation conv: 10 (from example)
      - Final conv filters: 2 for classification
      - Kernel size: default (3,3,3) for dilation conv, (1,1,1) for final conv
      - Activation: ReLU after batch norm
      - Use batch norm in dilation conv
      - Weight standardization to stabilize training with dilation

    This model is compatible with TF 2.20.0 and supports tf.function jit compilation.

    """

    def __init__(self, n_filters=10, dilation_rate=(3,3,3), wd=None, stddev=0.01, bn=True):
        super().__init__()
        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.wd = wd
        self.stddev = stddev
        self.bn = bn

        # Weight regularizer (L2)
        if wd is not None:
            self.kernel_regularizer = tf.keras.regularizers.l2(wd)
        else:
            self.kernel_regularizer = None

        # Dilated Conv3D with Weight Standardization
        self.dilation_conv = Conv3DWS(
            filters=self.n_filters,
            kernel_size=(3,3,3),
            strides=(1,1,1),
            padding='valid',   # Use 'valid' to avoid CUDNN issues with dilation + SAME
            dilation_rate=self.dilation_rate,
            activation=None,
            kernel_regularizer=self.kernel_regularizer,
            name='c1_dilation_conv'
        )

        if self.bn:
            self.batchnorm = tf.keras.layers.BatchNormalization(name='c1_bn')
        else:
            self.batchnorm = None

        # Final 1x1x1 convolution layer producing logits for 2 classes
        self.score_conv = tf.keras.layers.Conv3D(
            filters=2,
            kernel_size=(1,1,1),
            strides=(1,1,1),
            padding='valid',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=self.stddev),
            kernel_regularizer=self.kernel_regularizer,
            name='score_conv'
        )


    def call(self, inputs, training=False):
        # inputs: float32 tensor [B,H,W,D,C]
        x = self.dilation_conv(inputs)
        if self.bn is not None:
            # Use training flag for batch norm
            x = self.batchnorm(x, training=training)
        x = tf.nn.relu(x)

        score = self.score_conv(x)  # logits, no activation

        pred = tf.nn.softmax(score, axis=-1)

        # Return logits and softmax probabilities
        return score, pred


def my_model_function():
    # Return an instance of MyModel with example parameters
    # - similar settings as discussed (filters=10, dilation=(3,3,3))
    # Weight decay and stddev defaulted to None and 0.01 respectively
    return MyModel()


def GetInput():
    # Generate a random input tensor that fits the model's expected input shape
    # Based on test snippet: batch=1, height=140, width=140, depth=40, channels=1
    return tf.random.normal((1, 140, 140, 40, 1))


# --- Supporting class adapted from Chunk 7/7 for Weight Standardization ---

class Conv3DWS(tf.keras.layers.Conv3D):
    """
    Conv3D layer with Weight Standardization as per the provided example.

    Weight Standardization (WS) normalizes the convolution kernel weights to have zero mean and unit variance 
    channel-wise before applying convolution.

    This helps training stability especially for dilated convolutions and avoids some kernel issues.
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1,1,1), padding='same',
                 dilation_rate=(1,1,1), activation='relu', kernel_regularizer=None, name=''):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=None,  # We'll apply activation manually
            kernel_regularizer=kernel_regularizer,
            use_bias=True,
            name=name
        )
        # Activation function to apply after convolution + bias add
        if activation is not None:
            self.activation_fn = tf.keras.activations.get(activation)
        else:
            self.activation_fn = None

        # Data format for bias_add
        self._tf_data_format = 'NDHWC'  # TensorFlow default data format for Conv3D

    def call(self, x):
        # Weight standardization before convolution

        kernel_mean = tf.math.reduce_mean(self.kernel, axis=[0,1,2,3], keepdims=True, name='kernel_mean')
        kernel_std = tf.math.reduce_std(self.kernel, axis=[0,1,2,3], keepdims=True, name='kernel_std')
        kernel_new = (self.kernel - kernel_mean) / (kernel_std + tf.keras.backend.epsilon())

        # Compute strides and dilations in format required by tf.nn.conv3d
        strides_tf = [1] + list(self.strides) + [1]
        dilations_tf = [1] + list(self.dilation_rate) + [1]

        # Perform convolution with WS kernel
        output = tf.nn.conv3d(
            input=x,
            filters=kernel_new,
            strides=strides_tf,
            padding=self.padding.upper(),
            dilations=dilations_tf,
            data_format='NDHWC'
        )

        # Add bias if used
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format=self._tf_data_format)

        # Apply activation if set
        if self.activation_fn is not None:
            return self.activation_fn(output)
        else:
            return output

