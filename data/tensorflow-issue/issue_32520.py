import tensorflow as tf
import random
import os
import pdb
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import time
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random

dataset_path = "./"
train_labels_file = "dataset.txt"

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3
BATCH_SIZE = 128
NUM_ITERATIONS = 1000
#NUM_ITERATIONS = 10
LEARNING_RATE = 0.001
SUMMARY_LOG_DIR="./summary-log"
lasttime = 0
#IMAGE_HEIGHT = 100
#IMAGE_HEIGHT = 224
#IMAGE_WIDTH = 100
#IMAGE_WIDTH = 224
#BATCH_SIZE = 25
#NUM_CHANNELS = 3
#LEARNING_RATE = 0.0001
BATCH_SIZE = 128
OUTPUT=4096
NUM_CLASSES = OUTPUT


class MN_REDUCED(object):

    def __init__(self, trainable=True, dropout=0.5):
        self.trainable = trainable
        self.dropout = dropout
        self.parameters = []


    def build(self,rgb,train_mode=None):
        with tf.name_scope('conv_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, OUTPUT], dtype=tf.float32,
                                    stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                     trainable=True, name='biases')
            conv1 = tf.nn.bias_add(conv, biases)
         #   shape = int(np.prod(out.get_shape()))
         #   flat = tf.reshape(out, [BATCH_SIZE, -1])
         #   self.out_0 = flat[:,0:OUTPUT]

        with tf.name_scope('conv_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                    stddev=1e-2), name='weights')
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                     trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            shape = int(np.prod(out.get_shape()))
            flat = tf.reshape(out, [BATCH_SIZE, -1])
            self.out_0 = flat[:,0:OUTPUT]

    def loss(self, labels):
        labels = tf.cast(labels, tf.int32)
        oneHot = tf.one_hot (labels, NUM_CLASSES)
        loss = tf.reduce_mean(tf.square(self.out_0 - oneHot), name='loss')
        return loss

    def training(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss);
        return train_op


def placeholder_inputs(batch_size):
        images_placeholder = tf.placeholder(tf.float32,
                                                                shape=(batch_size, IMAGE_HEIGHT,
                                                                           IMAGE_WIDTH, NUM_CHANNELS))
        labels_placeholder = tf.placeholder(tf.int32,
                                                                shape=(batch_size))

        return images_placeholder, labels_placeholder

def fill_feed_dict(images_pl, labels_pl, sess):
        #images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])


        #feed_dict = {
        #       images_pl: images_feed,
        #       labels_pl: labels_feed,
        #}
        n = BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
        k = IMAGE_WIDTH
        a = np.empty(n, dtype=np.float32)
        np.random.seed(0)

        for i in range(0, n, k):
            a[i:i+k] = np.random.normal(loc=0, scale=1, size=k)
        rand = np.reshape(a, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))


        n1 = BATCH_SIZE
        a1 = np.empty(n1, dtype=np.float32)
        for i in range(0, n1):
            a1[i] = np.random.normal(loc=0, scale=1)


        feed_dict = {
                images_pl: rand,
                labels_pl: a1,
        }

        return feed_dict

def do_eval(sess,
                        eval_correct,
                        logits,
                        images_placeholder,
                        labels_placeholder,
                        dataset):

        true_count = 0
        # // is flooring division
        steps_per_epoch = dataset.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        for step in xrange(steps_per_epoch):
                #feed_dict = fill_feed_dict(dataset, images_placeholder,        labels_placeholder)
                feed_dict = fill_feed_dict(dataset, images_placeholder, labels_placeholder,sess)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                true_count = true_count + count

        precision = float(true_count) / num_examples
        print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                        (num_examples, true_count, precision))

def evaluation(logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        pred = tf.argmax(logits, 1)

        return tf.reduce_sum(tf.cast(correct, tf.int32))


def myTimer( str,iteration ):
        global lasttime
        start = time.time()
        elapse = start -lasttime
        lasttime = time.time()
        print ("%s iteration: %d elapse: %f" % (str,iteration, elapse))
        return;


def main():
        with tf.Graph().as_default():

                #data_input = DataInput(dataset_path, train_labels_file, BATCH_SIZE)
                images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
                cnn_full_cpu = MN_REDUCED()
                cnn_full_cpu.build(images_placeholder)

                summary = tf.summary.merge_all()
                saver = tf.train.Saver()
                sess = tf.Session()
                #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                loss = cnn_full_cpu.loss(labels_placeholder)
                train_op = cnn_full_cpu.training(loss)

                init = tf.global_variables_initializer()
                sess.run(init)
                eval_correct = evaluation(cnn_full_cpu.out_0, labels_placeholder)

                try:
                        for i in range(NUM_ITERATIONS):
                                feed_dict = fill_feed_dict(images_placeholder,
                                                                labels_placeholder, sess)

                                myTimer("start training", i)
                                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                                myTimer("end training", i)

                                print ('Step %d: loss = %.6f' % (i, loss_value))

                        coord.request_stop()
                        coord.join(threads)
                except Exception as e:
                        print(e)
                myTimer("start inference", i)
                infer = sess.run([cnn_full_cpu.out_0], feed_dict=feed_dict)
                myTimer("end inference", i)
        sess.close()

if __name__ == '__main__':
        main()