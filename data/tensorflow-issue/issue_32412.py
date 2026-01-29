# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← assuming CIFAR-10 images: (batch, 32, 32, 3)

import tensorflow as tf

class ConvBn2D(tf.keras.Model):
    def __init__(self, c_out, kernel_size=3):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=c_out,
            kernel_size=kernel_size,
            strides=1,
            padding="SAME",
            use_bias=False,
        )
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-7)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class FNet(tf.keras.Model):
    def __init__(self, start_kernels=64, weight=0.125, dynamic=False, **kwargs):
        super().__init__(**kwargs)
        c = start_kernels
        self.dynamic = dynamic

        self.max_pool = tf.keras.layers.MaxPooling2D()
        self.init_conv_bn = ConvBn2D(c, kernel_size=3)
        self.c0 = ConvBn2D(c, kernel_size=3)

        self.c1 = ConvBn2D(c * 2, kernel_size=3)
        self.c2 = ConvBn2D(c * 2, kernel_size=3)

        self.c3 = ConvBn2D(c * 2, kernel_size=3)
        self.c4 = ConvBn2D(c * 2, kernel_size=3)

        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.linear = tf.keras.layers.Dense(10, use_bias=False)
        self.weight = weight

    def call(self, x, training=False):
        h = self.init_conv_bn(x, training=training)
        h = self.c0(h, training=training)
        h = self.max_pool(h)

        h = self.c1(h, training=training)
        h = self.c2(h, training=training)
        h = self.max_pool(h)

        h = self.c3(h, training=training)
        h = self.c4(h, training=training)
        h = self.max_pool(h)

        h = self.pool(h)
        h = self.linear(h) * self.weight
        return h


class MyModel(tf.keras.Model):
    """
    Wrapper model fusing the example FNet model.
    """

    def __init__(self, start_kernels=8, dynamic=False):
        super().__init__()
        self.fnet = FNet(start_kernels=start_kernels, dynamic=dynamic)

    def call(self, inputs, training=False):
        # Just forward the FNet model
        return self.fnet(inputs, training=training)


def my_model_function():
    # Instantiate MyModel with example parameters reflecting those in the issue
    # dynamic=True was passed in the example code, so expose that here.
    return MyModel(start_kernels=8, dynamic=True)


def GetInput():
    # Returns a random CIFAR-10 like batch input: batch size 512 batches of 32x32 RGB images
    # dtype float32 to match the model expectations
    batch_size = 512
    height = 32
    width = 32
    channels = 3
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

