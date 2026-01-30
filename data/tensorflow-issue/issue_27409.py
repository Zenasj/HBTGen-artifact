def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        # with tf.variable_scope(name):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name


    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

CONV1_SIZE = 2
CONV1_DEPTH = 16

CONV2_SIZE = 2
CONV2_DEPTH = 32

CONV3_SIZE = 2
CONV3_DEPTH = 64

CONV4_SIZE = 2
CONV4_DEPTH = 128

FC_NODE = 512


f_dim = 64
channel_dim = 3

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# s_h = 252, 400
s_h, s_w = IMG_HEIGHT, IMG_WIDTH
print("s_h = ", s_h, "s_w = ", s_w)
# s_h2 = 126, s_w2 = 200
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
print("s_h2 = ", s_h2, "s_w2 = ", s_w2)
# s_h4 = 63, s_h4 = 100
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
print("s_h4 = ", s_h4, "s_w4 = ", s_w4)
# s_h8 = 32, s_h8 = 50
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
print("s_h8 = ", s_h8, "s_w8 = ", s_w8)
# s_h16 = 16, s_h16 = 25
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
print("s_h16 = ", s_h16, "s_w16 = ", s_w16)
print("=" * 20)

d_bn_0 = BatchNorm(name="d_bn_0")
d_bn_1 = BatchNorm(name="d_bn_1")
d_bn_2 = BatchNorm(name="d_bn_2")
d_bn_3 = BatchNorm(name="d_bn_3")
d_bn_4 = BatchNorm(name="d_bn_4")

c_bn_0 = BatchNorm(name="c_bn_0")


def inference(input_tensor, train, regularizer):

    with tf.variable_scope('input_reshape'):

        "convert the pressure data 2-D  to  4-D Tensor"
        z = linear(input_=input_tensor, output_size=f_dim * 8 * s_h16 * s_w16)
        h0 = tf.reshape(z, [-1, s_h16, s_w16, f_dim * 8])
        h0 = tf.nn.relu(d_bn_0(h0))
        # [None, 16, 25, 512]
        print("h0.shape", h0.get_shape().as_list())

        # h1 = upsampling(h0, s_h8, s_w8)
        h1 = deconv2d(input_=h0, output_shape=[batch_size, s_h8, s_w8, f_dim * 4], name="deconv1")
        h1 = d_bn_1(h1)
        # [None, 32, 50, 512]
        print("h1.shape", h1.get_shape().as_list())
        # h1 = conv2d(input_=h1, output_dim=f_dim * 4, d_h=1, d_w=1, name="conv_upsample1")
        # [None, 32, 50, 256]
        # print("h1.shape", h1.get_shape().as_list())
        # h2 = upsampling(h1, s_h4, s_w4)
        h2 = deconv2d(input_=h1, output_shape=[batch_size, s_h4, s_w4, f_dim * 2], name="deconv2")
        h2 = d_bn_2(h2)
        # [None, 63, 100, 256]
        print("h2.shape", h2.get_shape().as_list())
        # h2 = conv2d(input_=h2, output_dim=f_dim * 2, d_h=1, d_w=1, name="conv_upsample2")
        # # [None, 63, 100, 128]
        # print("h2.shape", h2.get_shape().as_list())
        # h3 = upsampling(h2, s_h2, s_w2)
        # [None, 126, 200, 128]
        h3 = deconv2d(input_=h2, output_shape=[batch_size, s_h2, s_w2, f_dim], name="deconv3")
        h3 = d_bn_3(h3)
        print("h3.shape", h3.get_shape().as_list())
        # h3 = conv2d(input_=h3, output_dim=f_dim, d_h=1, d_w=1, name="conv_upsample3")
        # # [None, 126, 200, 64]
        # print("h3.shape", h3.get_shape().as_list())
        # h4 = upsampling(h3, s_h, s_w)
        # [None, 252, 400, 64]
        h4 = deconv2d(input_=h3, output_shape=[batch_size, s_h, s_w, channel_dim], name="deconv4")
        h4 = d_bn_4(h4)
        # print("h4.shape", h4.get_shape().as_list())
        # h4 = conv2d(input_=h4, output_dim=f_dim//2, d_h=1, d_w=1, name="conv_upsample4")
        # # [None, 252, 400, 32]
        print("h4.shape", h4.get_shape().as_list())


    print("=" * 20)

    conv1 = c_bn_0(conv2d(input_=h4, output_dim=CONV1_DEPTH, name="conv1"))

    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(
            conv1, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL1 shape [64,63, 100, 16]
        print("POOL1 shape", pool1.get_shape().as_list())

    conv2 = conv2d(input_=pool1, output_dim=CONV2_DEPTH, name="conv2")

    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(
            conv2, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL2 shape [64, 16, 25, 32]
        print("POOL2 shape", pool2.get_shape().as_list())

    conv3 = conv2d(input_=pool2, output_dim=CONV3_DEPTH, name="conv3")

    with tf.variable_scope("pool3"):
        pool3 = tf.nn.max_pool(
            conv3, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL3 shape [64, 4, 7, 64]
        print("POOL3 shape", pool3.get_shape().as_list())

    conv4 = conv2d(input_=pool3, output_dim=CONV4_DEPTH, name="conv4")

    with tf.variable_scope("fullly_reshape"):
        pool_shape = conv4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # [batch_size, 2*4*128]
        reshaped = tf.reshape(conv4, [-1, nodes])
        # reshaped shape [64, 1024]
        print("reshaped shape", reshaped.get_shape().as_list())

    with tf.variable_scope("fully1"):

        with tf.name_scope("weights"):
            fc1_weights = tf.get_variable(
                name="fc1_weights",
                shape=[
                    nodes,
                    FC_NODE],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1))
            print("fc1_weights", fc1_weights.get_shape())
            variable_summaries(fc1_weights, "fully1" + "/weights")
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        with tf.name_scope("biases"):
            fc1_biases = tf.get_variable(
                name="fc1_biases",
                shape=[FC_NODE],
                initializer=tf.constant_initializer(0.1))
            variable_summaries(fc1_biases, "fully1" + "/biases")
        # [batch_size, FC_NODE]
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    with tf.variable_scope("fully2"):

        with tf.name_scope("weights"):
            fc2_weights = tf.get_variable(
                name="fc2_weights",
                shape=[
                    FC_NODE,
                    NUM_LABELS],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1))
            print("fc2_weights", fc2_weights.get_shape())
            variable_summaries(fc2_weights, "fully2" + "/weights")
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        with tf.name_scope("biases"):
            fc2_biases = tf.get_variable(
                name="fc2_biases",
                shape=[NUM_LABELS],
                initializer=tf.constant_initializer(0.1))
            variable_summaries(fc2_biases, "fully2" + "/biases")
        # [batch_size, 252*400*3]
        fc2 = tf.matmul(fc1, fc2_weights)+fc2_biases

    with tf.variable_scope("output_reshape"):
        output_array = tf.reshape(fc2, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        print("output_shape", output_array.get_shape().as_list())

    tf.summary.histogram("output_array", output_array)
    """[64, 252, 400, 3]"""
    return output_array

class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

def linear(input_, output_size, scope="Linear", stddev=0.1, bias_start=0.1, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        try:
            matrix = tf.get_variable(name="weights", shape=[shape[1], output_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            variable_summaries(matrix, scope+"weights")
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions. " \
                  " Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        variable_summaries(bias, scope+"bias")
        # print("with_w为False,只返回了矩阵相乘的结果")
        return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim,
        k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.1,
        name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        variable_summaries(w, name+"weights")
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, name+"biases")
        conv = tf.nn.relu(tf.nn.bias_add(conv, biases))
        print(name+".shape", conv.get_shape().as_list())

    return conv

def deconv2d(input_, output_shape,
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.1,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(name='w', shape=[k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        variable_summaries(w, name+"_weights")
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.1))
        variable_summaries(biases, name + "_bias")
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

import tensorflow as tf


def variable_summaries(var, name):

    # 将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope("summaries"):
        # 通过tf.summary.histogram记录张量中元素的取值分布
        tf.summary.histogram(name, var)
        # 计算变量的平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算变量的标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def upsampling(x, height, width):
    # 临界点插值进行图片的放大，上采样
    with tf.name_scope("image.resize"):
        return tf.image.resize_nearest_neighbor(x, [height, width])


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name


    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)

def linear(input_, output_size, scope="Linear", stddev=0.1, bias_start=0.1, with_w=False):

    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        try:
            matrix = tf.get_variable(name="weights", shape=[shape[1], output_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            variable_summaries(matrix, scope+"weights")
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image dimensions. " \
                  " Did you correctly set '--crop' or '--input_height' or '--output_height'?"
            err.args = err.args + (msg,)
            raise
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        variable_summaries(bias, scope+"bias")
        # print("with_w为False,只返回了矩阵相乘的结果")
        return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim,
        k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.1,
        name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        variable_summaries(w, name+"weights")
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, name+"biases")
        conv = tf.nn.relu(tf.nn.bias_add(conv, biases))
        print(name+".shape", conv.get_shape().as_list())

    return conv

def deconv2d(input_, output_shape,
             k_h=2, k_w=2, d_h=2, d_w=2, stddev=0.1,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable(name='w', shape=[k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        variable_summaries(w, name+"_weights")
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.1))
        variable_summaries(biases, name + "_bias")
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

import tensorflow as tf
import math
from ops_CNN import *
from train_read_from_tfrecords import batch_size
from tfrecords_train import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL


NUM_LABELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNEL

# "学不到比较细小的特征，是不是我这个卷积核的尺寸给大了，局部感受野太大，学不到细小的特征？？？"

CONV1_SIZE = 2
CONV1_DEPTH = 16

CONV2_SIZE = 2
CONV2_DEPTH = 32

CONV3_SIZE = 2
CONV3_DEPTH = 64

CONV4_SIZE = 2
CONV4_DEPTH = 128

FC_NODE = 512


f_dim = 64
channel_dim = 3

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# s_h = 252, 400
s_h, s_w = IMG_HEIGHT, IMG_WIDTH
print("s_h = ", s_h, "s_w = ", s_w)
# s_h2 = 126, s_w2 = 200
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
print("s_h2 = ", s_h2, "s_w2 = ", s_w2)
# s_h4 = 63, s_h4 = 100
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
print("s_h4 = ", s_h4, "s_w4 = ", s_w4)
# s_h8 = 32, s_h8 = 50
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
print("s_h8 = ", s_h8, "s_w8 = ", s_w8)
# s_h16 = 16, s_h16 = 25
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
print("s_h16 = ", s_h16, "s_w16 = ", s_w16)
print("=" * 20)

d_bn_0 = BatchNorm(name="d_bn_0")
d_bn_1 = BatchNorm(name="d_bn_1")
d_bn_2 = BatchNorm(name="d_bn_2")
d_bn_3 = BatchNorm(name="d_bn_3")
d_bn_4 = BatchNorm(name="d_bn_4")

c_bn_0 = BatchNorm(name="c_bn_0")


def inference(input_tensor, train, regularizer):

    with tf.variable_scope('input_reshape'):

        "convert the pressure data 2-D  to  4-D Tensor"
        z = linear(input_=input_tensor, output_size=f_dim * 8 * s_h16 * s_w16)
        h0 = tf.reshape(z, [-1, s_h16, s_w16, f_dim * 8])
        h0 = tf.nn.relu(d_bn_0(h0))
        # [None, 16, 25, 512]
        print("h0.shape", h0.get_shape().as_list())

        # h1 = upsampling(h0, s_h8, s_w8)
        h1 = deconv2d(input_=h0, output_shape=[batch_size, s_h8, s_w8, f_dim * 4], name="deconv1")
        h1 = d_bn_1(h1)
        # [None, 32, 50, 512]
        print("h1.shape", h1.get_shape().as_list())
        # h1 = conv2d(input_=h1, output_dim=f_dim * 4, d_h=1, d_w=1, name="conv_upsample1")
        # [None, 32, 50, 256]
        # print("h1.shape", h1.get_shape().as_list())
        # h2 = upsampling(h1, s_h4, s_w4)
        h2 = deconv2d(input_=h1, output_shape=[batch_size, s_h4, s_w4, f_dim * 2], name="deconv2")
        h2 = d_bn_2(h2)
        # [None, 63, 100, 256]
        print("h2.shape", h2.get_shape().as_list())
        # h2 = conv2d(input_=h2, output_dim=f_dim * 2, d_h=1, d_w=1, name="conv_upsample2")
        # # [None, 63, 100, 128]
        # print("h2.shape", h2.get_shape().as_list())
        # h3 = upsampling(h2, s_h2, s_w2)
        # [None, 126, 200, 128]
        h3 = deconv2d(input_=h2, output_shape=[batch_size, s_h2, s_w2, f_dim], name="deconv3")
        h3 = d_bn_3(h3)
        print("h3.shape", h3.get_shape().as_list())
        # h3 = conv2d(input_=h3, output_dim=f_dim, d_h=1, d_w=1, name="conv_upsample3")
        # # [None, 126, 200, 64]
        # print("h3.shape", h3.get_shape().as_list())
        # h4 = upsampling(h3, s_h, s_w)
        # [None, 252, 400, 64]
        h4 = deconv2d(input_=h3, output_shape=[batch_size, s_h, s_w, channel_dim], name="deconv4")
        h4 = d_bn_4(h4)
        # print("h4.shape", h4.get_shape().as_list())
        # h4 = conv2d(input_=h4, output_dim=f_dim//2, d_h=1, d_w=1, name="conv_upsample4")
        # # [None, 252, 400, 32]
        print("h4.shape", h4.get_shape().as_list())


    print("=" * 20)

    conv1 = c_bn_0(conv2d(input_=h4, output_dim=CONV1_DEPTH, name="conv1"))

    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(
            conv1, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL1 shape [64,63, 100, 16]
        print("POOL1 shape", pool1.get_shape().as_list())

    conv2 = conv2d(input_=pool1, output_dim=CONV2_DEPTH, name="conv2")

    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(
            conv2, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL2 shape [64, 16, 25, 32]
        print("POOL2 shape", pool2.get_shape().as_list())

    conv3 = conv2d(input_=pool2, output_dim=CONV3_DEPTH, name="conv3")

    with tf.variable_scope("pool3"):
        pool3 = tf.nn.max_pool(
            conv3, ksize=[
                1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding="SAME")
        # POOL3 shape [64, 4, 7, 64]
        print("POOL3 shape", pool3.get_shape().as_list())

    conv4 = conv2d(input_=pool3, output_dim=CONV4_DEPTH, name="conv4")

    with tf.variable_scope("fullly_reshape"):
        pool_shape = conv4.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # [batch_size, 2*4*128]
        reshaped = tf.reshape(conv4, [-1, nodes])
        # reshaped shape [64, 1024]
        print("reshaped shape", reshaped.get_shape().as_list())

    with tf.variable_scope("fully1"):

        with tf.name_scope("weights"):
            fc1_weights = tf.get_variable(
                name="fc1_weights",
                shape=[
                    nodes,
                    FC_NODE],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1))
            print("fc1_weights", fc1_weights.get_shape())
            variable_summaries(fc1_weights, "fully1" + "/weights")
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        with tf.name_scope("biases"):
            fc1_biases = tf.get_variable(
                name="fc1_biases",
                shape=[FC_NODE],
                initializer=tf.constant_initializer(0.1))
            variable_summaries(fc1_biases, "fully1" + "/biases")
        # [batch_size, FC_NODE]
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    with tf.variable_scope("fully2"):

        with tf.name_scope("weights"):
            fc2_weights = tf.get_variable(
                name="fc2_weights",
                shape=[
                    FC_NODE,
                    NUM_LABELS],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1))
            print("fc2_weights", fc2_weights.get_shape())
            variable_summaries(fc2_weights, "fully2" + "/weights")
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc2_weights))

        with tf.name_scope("biases"):
            fc2_biases = tf.get_variable(
                name="fc2_biases",
                shape=[NUM_LABELS],
                initializer=tf.constant_initializer(0.1))
            variable_summaries(fc2_biases, "fully2" + "/biases")
        # [batch_size, 252*400*3]
        fc2 = tf.matmul(fc1, fc2_weights)+fc2_biases

    with tf.variable_scope("output_reshape"):
        output_array = tf.reshape(fc2, [-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        print("output_shape", output_array.get_shape().as_list())

    tf.summary.histogram("output_array", output_array)
    """[64, 252, 400, 3]"""
    return output_array

import tensorflow as tf
from tfrecords_train import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL

buffer_size = 100000
batch_size = 32
num_epochs = 500
filename = ["train.tfrecords"]


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            "images_raw": tf.FixedLenFeature([], tf.string),
            "features": tf.FixedLenFeature([10], tf.float32)
        }
    )

    # "这里解析的时候图像的数据类型一定要是uint8!!!!!"
    decoded_image = tf.decode_raw(features['images_raw'], tf.uint8)
    decoded_image = tf.reshape(decoded_image, [IMG_HEIGHT,IMG_WIDTH,IMG_CHANNEL])
    # decoded_images.shape [252, 400, 3]
    # "但是可以在这里进行数据类型的修改！！！"
    decoded_image = tf.cast(decoded_image, tf.float32)
    print("decoded_images.shape", decoded_image.get_shape().as_list())

    feature = features['features']
    # feature.shape [10]
    print("feature.shape", feature.get_shape().as_list())
    return decoded_image, feature


with tf.name_scope('input_data') as scope:
    Dataset = tf.data.TFRecordDataset(filename)
    print("=" * 10)
    print("batch_size=", batch_size)
    Dataset = Dataset.map(parser).shuffle(buffer_size=buffer_size).batch(batch_size=batch_size).repeat(num_epochs)
    train_iterator = Dataset.make_initializable_iterator()
    train_next_element = train_iterator.get_next()