# tf.random.uniform((1024, 28, 28, 1), dtype=tf.float32) ← inferred from the input shape used in the example with batch_size=1024

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initial Conv2D(32, 5x5, stride=2) layer
        self.initial_conv = tf.keras.layers.Conv2D(32, 5, strides=2, use_bias=False, padding="SAME")

        # First dense block repeated 5 times (with bottleneck conv 1x1 and conv 3x3)
        self.dense_block1_bn1 = [tf.keras.layers.BatchNormalization() for _ in range(5)]
        self.dense_block1_relu1 = [tf.keras.layers.ReLU() for _ in range(5)]
        self.dense_block1_conv1 = [tf.keras.layers.Conv2D(16, 1, use_bias=False, padding="SAME") for _ in range(5)]
        self.dense_block1_bn2 = [tf.keras.layers.BatchNormalization() for _ in range(5)]
        self.dense_block1_relu2 = [tf.keras.layers.ReLU() for _ in range(5)]
        self.dense_block1_conv2 = [tf.keras.layers.Conv2D(4, 3, use_bias=False, padding="SAME") for _ in range(5)]

        # Transition layers after first block
        self.trans1_bn = tf.keras.layers.BatchNormalization()
        self.trans1_relu = tf.keras.layers.ReLU()
        self.trans1_conv = tf.keras.layers.Conv2D(64, 1, use_bias=False, padding="SAME")
        self.trans1_pool = tf.keras.layers.AveragePooling2D()

        # Second dense block repeated 10 times
        self.dense_block2_bn1 = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        self.dense_block2_relu1 = [tf.keras.layers.ReLU() for _ in range(10)]
        self.dense_block2_conv1 = [tf.keras.layers.Conv2D(32, 1, use_bias=False, padding="SAME") for _ in range(10)]
        self.dense_block2_bn2 = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        self.dense_block2_relu2 = [tf.keras.layers.ReLU() for _ in range(10)]
        self.dense_block2_conv2 = [tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="SAME") for _ in range(10)]

        # Transition layers after second block
        self.trans2_bn = tf.keras.layers.BatchNormalization()
        self.trans2_relu = tf.keras.layers.ReLU()
        self.trans2_conv = tf.keras.layers.Conv2D(128, 1, use_bias=False, padding="SAME")
        self.trans2_pool = tf.keras.layers.AveragePooling2D()

        # Third dense block repeated 10 times (same as second block)
        self.dense_block3_bn1 = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        self.dense_block3_relu1 = [tf.keras.layers.ReLU() for _ in range(10)]
        self.dense_block3_conv1 = [tf.keras.layers.Conv2D(32, 1, use_bias=False, padding="SAME") for _ in range(10)]
        self.dense_block3_bn2 = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        self.dense_block3_relu2 = [tf.keras.layers.ReLU() for _ in range(10)]
        self.dense_block3_conv2 = [tf.keras.layers.Conv2D(8, 3, use_bias=False, padding="SAME") for _ in range(10)]

        # Final layers
        self.final_bn = tf.keras.layers.BatchNormalization()
        self.final_relu = tf.keras.layers.ReLU()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_out = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Softmax()

        # Wrap blocks by recompute_grad if flagged.
        # As we do not have a flag here, we just define recompute_grad as identity function.
        # In real use, it would be something like:
        # from tensorflow.python.ops.custom_gradient import recompute_grad
        # But due to the issue, we leave it disabled here.
        self.use_recompute_grad = False

    def dense_block1(self, x, i):
        # Define the _block from the snippet for first dense block i-th repetition
        def block(x_in):
            x_in = self.dense_block1_bn1[i](x_in)
            x_in = self.dense_block1_relu1[i](x_in)
            x_in = self.dense_block1_conv1[i](x_in)
            x_in = self.dense_block1_bn2[i](x_in)
            x_in = self.dense_block1_relu2[i](x_in)
            x_in = self.dense_block1_conv2[i](x_in)
            return x_in

        if self.use_recompute_grad:
            block = tf.recompute_grad(block)
        return block(x)

    def dense_block2(self, x, i):
        # Second dense block per repetition
        def block(x_in):
            x_in = self.dense_block2_bn1[i](x_in)
            x_in = self.dense_block2_relu1[i](x_in)
            x_in = self.dense_block2_conv1[i](x_in)
            x_in = self.dense_block2_bn2[i](x_in)
            x_in = self.dense_block2_relu2[i](x_in)
            x_in = self.dense_block2_conv2[i](x_in)
            return x_in

        if self.use_recompute_grad:
            block = tf.recompute_grad(block)
        return block(x)

    def dense_block3(self, x, i):
        # Third dense block per repetition, similar to second block
        def block(x_in):
            x_in = self.dense_block3_bn1[i](x_in)
            x_in = self.dense_block3_relu1[i](x_in)
            x_in = self.dense_block3_conv1[i](x_in)
            x_in = self.dense_block3_bn2[i](x_in)
            x_in = self.dense_block3_relu2[i](x_in)
            x_in = self.dense_block3_conv2[i](x_in)
            return x_in

        if self.use_recompute_grad:
            block = tf.recompute_grad(block)
        return block(x)

    def call(self, inputs, training=False):
        net = self.initial_conv(inputs)

        # First dense block (5 layers)
        for i in range(5):
            out = self.dense_block1(net, i)
            net = tf.keras.layers.concatenate([net, out])

        net = self.trans1_bn(net, training=training)
        net = self.trans1_relu(net)
        net = self.trans1_conv(net)
        net = self.trans1_pool(net)

        # Second dense block (10 layers)
        for i in range(10):
            out = self.dense_block2(net, i)
            net = tf.keras.layers.concatenate([net, out])

        net = self.trans2_bn(net, training=training)
        net = self.trans2_relu(net)
        net = self.trans2_conv(net)
        net = self.trans2_pool(net)

        # Third dense block (10 layers)
        for i in range(10):
            out = self.dense_block3(net, i)
            net = tf.keras.layers.concatenate([net, out])

        net = self.final_bn(net, training=training)
        net = self.final_relu(net)
        net = self.global_avg_pool(net)
        net = self.dense_out(net)
        net = self.softmax(net)
        return net


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor compatible with MyModel input
    # Batch size 1024 (matches the example), 28x28 grayscale images with 1 channel
    return tf.random.uniform((1024, 28, 28, 1), dtype=tf.float32)

