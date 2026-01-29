# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred as batches of 28x28 grayscale images

import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        K=1,
        backbone="residual",
        **kwargs
    ):
        super(ConvBlock, self).__init__(**kwargs)
        # Conv layer 'f' with relu activation and batch norm before it
        self.bn_f1 = tf.keras.layers.BatchNormalization()
        self.f = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=kernel_size,
            strides=stride,
            padding="SAME",
            use_bias=bias,
            activation=None,
        )
        # Conv layer 'g', always 3x3, stride 1, no bias
        # Will add batch norm inside call due to original code's dynamic creation (inferred here as a permanent layer)
        self.bn_g = tf.keras.layers.BatchNormalization()
        self.g = tf.keras.layers.Conv2D(
            c_out,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            activation=None,
        )
        self.K = K
        self.backbone = backbone
        self.c_out = c_out

    def call(self, x, training=False):
        # Apply batch norm + relu -> conv 'f'
        x1 = self.bn_f1(x, training=training)
        x1 = tf.nn.relu(x1)
        f = self.f(x1)

        # Apply BN + relu -> conv 'g' on f output
        h = self.bn_g(f, training=training)
        h = tf.nn.relu(h)
        h = self.g(h)
        return h


class MyModel(tf.keras.Model):
    def __init__(self, backbone="cnn", K=5):
        super(MyModel, self).__init__()
        self.backbone = backbone
        self.K = K

        # Input shape is fixed: (28, 28, 1)
        # Use ConvBlock from provided code: input 1 channel, output 1 channel for demo
        self.conv = ConvBlock(1, 1, kernel_size=3, K=K, backbone=backbone)
        self.relu = tf.keras.layers.Activation("relu")
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # inputs expected shape: (batch_size, 28, 28, 1)
        x = inputs
        x = self.conv(x, training=training)
        f1 = x  # output after conv block
        x = self.relu(x)
        x = self.avg_pool(x)
        f2 = x  # output after avg pooling
        x = self.flatten(x)
        out = self.fc(x)

        # Return outputs tuple as in original code: classification + two intermediate features
        return out, f1, f2


def my_model_function():
    # Instantiate and return the MyModel instance with default parameters
    return MyModel()


def GetInput():
    # Return a random tensor input with batch dimension 1 and shape (28, 28, 1)
    # Using float32 as dtype common for TF models
    return tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

