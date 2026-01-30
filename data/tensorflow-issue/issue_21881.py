#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset


class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 32, 32, 3), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        image = tf.transpose(image, [0, 3, 1, 2])
        data_format = 'channels_first'

        with argscope(Conv2D, activation=BNReLU, use_bias=False, kernel_size=3), \
                argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format=data_format):
            logits = LinearWrap(image) \
                .Conv2D('conv1.1', filters=66) \
                .Conv2D('conv1.2', filters=128) \
                .Conv2D('conv2.1', filters=128) \
                .Conv2D('conv2.2', filters=128) \
                .Conv2D('conv2.3', filters=192) \
                .MaxPooling('pool2', 2, stride=2, padding='SAME') \
                .tf.nn.dropout(0.95) \
                .Conv2D('conv4.1', filters=192) \
                .Conv2D('conv4.2', filters=192) \
                .Conv2D('conv4.3', filters=192) \
                .Conv2D('conv4.4', filters=192) \
                .Conv2D('conv4.5', filters=288) \
                .MaxPooling('pool2', 2, stride=2, padding='SAME') \
                .tf.nn.dropout(0.95) \
                .Conv2D('conv5.1', filters=288) \
                .Conv2D('conv5.2', filters=355) \
                .Conv2D('conv5.3', filters=432) \
                .GlobalAvgPooling('gap') \
                .FullyConnected('linear', out_dim=self.cifar_classnum)()

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.to_float(tf.nn.in_top_k(logits, label, 1), name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((32, 32)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 100, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds


def get_config(cifar_classnum):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    dataset_test = get_data('test', cifar_classnum)
    return TrainConfig(
        model=Model(cifar_classnum),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            ScalarStats(['accuracy', 'cost'])),
        ],
        max_epoch=150,
    )


if __name__ == '__main__':
    with tf.Graph().as_default():
        logger.set_logger_dir(os.path.join('train_log', 'cifar' + str(10)))
        config = get_config(10)

        trainer = SimpleTrainer()
        launch_train_with_config(config, trainer)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset
import tflearn

class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 32, 32, 3), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        image = tf.transpose(image, [0, 3, 1, 2])
        data_format = 'channels_first'

        with argscope(Conv2D, activation=BNReLU, use_bias=True, kernel_size=3), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format=data_format):
            logits = LinearWrap(image) \
                .Conv2D('conv1.1', filters=66) \
                .Conv2D('conv1.2', filters=128) \
                .Conv2D('conv2.1', filters=128) \
                .Conv2D('conv2.2', filters=128) \
                .Conv2D('conv2.3', filters=192) \
                .MaxPooling('pool2', 2, stride=2, padding='SAME') \
                .tf.nn.dropout(0.95) \
                .Conv2D('conv4.1', filters=192) \
                .Conv2D('conv4.2', filters=192) \
                .Conv2D('conv4.3', filters=192) \
                .Conv2D('conv4.4', filters=192) \
                .Conv2D('conv4.5', filters=288) \
                .MaxPooling('pool2', 2, stride=2, padding='SAME') \
                .tf.nn.dropout(0.95) \
                .Conv2D('conv5.1', filters=288) \
                .Conv2D('conv5.2', filters=355) \
                .Conv2D('conv5.3', filters=432)() 
            logits = tflearn.layers.conv.global_max_pool (logits, name='GlobalMaxPool')
            logits = LinearWrap(logits) \
                .FullyConnected('linear', out_dim=self.cifar_classnum)()

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.to_float(tf.nn.in_top_k(logits, label, 1), name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.GradientDescentOptimizer(lr)


def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((32, 32)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 100, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds


def get_config(cifar_classnum):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    dataset_test = get_data('test', cifar_classnum)
    return TrainConfig(
        model=Model(cifar_classnum),
        data=QueueInput(dataset_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            ScalarStats(['accuracy', 'cost'])),
        ],
        max_epoch=150,
    )


if __name__ == '__main__':
    with tf.Graph().as_default():
        logger.set_logger_dir(os.path.join('train_log', 'cifar' + str(10)))
        config = get_config(10)

        trainer = SimpleTrainer()
        launch_train_with_config(config, trainer)