# tf.random.uniform((128, 32, 32, 3), dtype=tf.float32)

import tensorflow as tf
import numpy as np
import time

# Constants inferred from the original code and context
BATCH_SIZE = 128
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3
OUTPUT = 4096
NUM_CLASSES = OUTPUT


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two conv layers similar to MN_REDUCED.build()
        # We mimic variable initialization and conv2d parameters
        self.conv1_kernel = self.add_weight(
            name="conv1_weights",
            shape=(3, 3, NUM_CHANNELS, OUTPUT),
            initializer=tf.random.truncated_normal_initializer(stddev=1e-2),
            trainable=True,
        )
        self.conv1_biases = self.add_weight(
            name="conv1_biases",
            shape=(OUTPUT,),
            initializer=tf.zeros_initializer(),
            trainable=True,
        )

        self.conv2_kernel = self.add_weight(
            name="conv2_weights",
            shape=(3, 3, OUTPUT, OUTPUT),
            initializer=tf.random.truncated_normal_initializer(stddev=1e-2),
            trainable=True,
        )
        self.conv2_biases = self.add_weight(
            name="conv2_biases",
            shape=(OUTPUT,),
            initializer=tf.zeros_initializer(),
            trainable=True,
        )

    def call(self, inputs, training=False):
        # inputs shape: [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]
        # Apply conv1
        conv1 = tf.nn.conv2d(inputs, self.conv1_kernel, strides=[1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.bias_add(conv1, self.conv1_biases)

        # Apply conv2
        conv2 = tf.nn.conv2d(conv1, self.conv2_kernel, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, self.conv2_biases)

        # Flatten conv2 output and slice first OUTPUT elements for output
        batch_flat = tf.reshape(conv2, [BATCH_SIZE, -1])
        out = batch_flat[:, 0:OUTPUT]

        return out

    def loss_fn(self, logits, labels):
        # labels expected as floats from original code, but cast to int32 internally
        labels = tf.cast(labels, tf.int32)
        one_hot = tf.one_hot(labels, NUM_CLASSES)
        loss = tf.reduce_mean(tf.square(logits - one_hot), name="loss")
        return loss


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Produce a tensor shaped (128, 32, 32, 3) with float32 values,
    # random normal distributed as in fill_feed_dict. Mimic RNG seeding and shape carefulness.
    np.random.seed(0)

    n = BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
    k = IMAGE_WIDTH
    a = np.empty(n, dtype=np.float32)

    for i in range(0, n, k):
        a[i:i + k] = np.random.normal(loc=0, scale=1, size=k)
    # reshape a in [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS] in original code
    # Note: original code used (B, W, H, C) order instead of standard NHWC (B,H,W,C)
    # The original code swap width/height in np.reshape, but convolution expects NHWC
    # Here we keep NHWC by reshaping accordingly:
    # chunk says: rand = np.reshape(a, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))
    # That means width and height positions swapped relative to standard NHWC,
    # we keep their order as is to match original code.
    input_np = np.reshape(a, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)).astype(np.float32)

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(input_np)

    return input_tensor

