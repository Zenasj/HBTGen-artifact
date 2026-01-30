import random

import numpy
import tensorflow as tf


class Model_1conv2d(tf.Module):
    def __init__(self, kernel):
        super().__init__()
        self.conv2d_weigths = tf.constant(kernel)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.float32, name='input')])
    def f(self, input_1):
        conv2d = tf.nn.conv2d(
            input_1,
            self.conv2d_weigths,
            data_format="NHWC",
            padding="SAME",
            dilations=[1, 1],
            strides=[1, 1, 1, 1],
            name="conv2d")
        return conv2d

class Model_2conv2d(tf.Module):
    def __init__(self, k1, k2):
        super().__init__()
        self.conv2d_weigths_1 = tf.constant(k1)
        self.conv2d_weigths_2 = tf.constant(k2)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(1, 512, 512, 3), dtype=tf.float32, name='input_1_input')
    ])
    def f(self, input_1):
        conv2d_1 = tf.nn.conv2d(
            input_1,
            self.conv2d_weigths_1,
            data_format="NHWC",
            padding="SAME",
            dilations=[1, 1],
            strides=[1, 1, 1, 1],
            name="conv2d1")
        conv2d_2 = tf.nn.conv2d(
            input_1,
            self.conv2d_weigths_2,
            data_format="NHWC",
            padding="SAME",
            dilations=[1, 1],
            strides=[1, 1, 1, 1],
            name="conv2d2")
        concat = tf.concat([
            conv2d_1,
            conv2d_2,],
            axis=3,
            name="output")
        return concat


def _test(depth):

    kernel = (numpy.random.uniform(
        low=-0.05,
        high=0.05,
        size=[3, 3, 3, depth])).astype(dtype=numpy.float16).astype(dtype=numpy.float32)

    x = 2 * numpy.random.rand(1, 512, 512, 3).astype(dtype=numpy.float32)
    input_1_feed = numpy.where(x > 1, x - 0.5, x - 1.5).astype(dtype=numpy.float32)

    model_1 = Model_1conv2d(kernel)
    model_2 = Model_2conv2d(kernel[:, :, :, :(depth // 2)], kernel[:, :, :, (depth // 2):])

    out_1 = model_1.f(input_1_feed,).numpy()
    out_2 = model_2.f(input_1_feed,).numpy()

    numpy.testing.assert_allclose(out_1, out_2)


if __name__ == '__main__':
    for d in [32, 8, 16]:
        _test(d)
        print('>> test with kernel depth = %2s pass' % d)