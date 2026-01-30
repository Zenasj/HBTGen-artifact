import math
from tensorflow import keras
from tensorflow.keras import layers

tensorflow-gpu

tf.keras

tf.keras.layers.BatchNormalization

training=True

training=False

training

import os

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.tools import freeze_graph


class Model:
    def __init__(self):
        self._conv = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=tf.nn.relu)
        self._batch_norm = tf.keras.layers.BatchNormalization()
        self._flatten = tf.keras.layers.Flatten()
        self._logits = tf.keras.layers.Dense(units=10)

    def forward(self, inputs):
        with tf.name_scope('model'):
            conv_out = self._conv(inputs)
            norm_out = self._batch_norm(conv_out)
            flat_out = self._flatten(norm_out)
            logits = self._logits(flat_out)

        return tf.identity(logits, name='logits')

    @staticmethod
    def loss(logits, labels):
        with tf.name_scope('loss'):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        return loss


def train():
    def map_fn(x, y):
        return tf.expand_dims(tf.math.divide(tf.cast(x, tf.float32), 255), axis=2), tf.cast(y, tf.int32)

    (x_train, y_train), (_, _) = mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.map(map_fn).batch(128).repeat(10)

    iterator = dataset.make_one_shot_iterator()
    inputs, labels = iterator.get_next()

    model = Model()
    logits = model.forward(inputs)
    loss = model.loss(logits, labels)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e4).minimize(loss,
                                                                       global_step=tf.train.get_or_create_global_step())

    with tf.train.MonitoredTrainingSession(checkpoint_dir='./tmp') as sess:
        while not sess.should_stop():
            print(sess.run([optimizer, loss])[1])


def export():
    tf.reset_default_graph()

    export_dir = './tmp/frozen'
    model = Model()

    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input_placeholder')
    _ = model.forward(inputs)

    with open('./tmp/checkpoint') as f:
        line = f.readline()
        checkpoint = line[line.find('"') + 1:line.rfind('"')]
        print(checkpoint)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        print(os.path.join('./tmp', checkpoint))
        saver.restore(sess, os.path.join('./tmp', checkpoint))

        builder = tf.saved_model.Builder('./tmp/frozen')
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
        builder.save()

        graph = tf.graph_util.remove_training_nodes(sess.graph.as_graph_def(), protected_nodes=['logits'])

        freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=graph,
            input_saver_def=None,
            input_saved_model_dir=export_dir,
            saved_model_tags=[tf.saved_model.tag_constants.SERVING],
            input_checkpoint=os.path.join('./tmp', checkpoint),
            output_node_names='logits',
            output_graph=os.path.join('./tmp', 'frozen.pb'),
            clear_devices=True,
            initializer_nodes='',
            restore_op_name='',
            filename_tensor_name='')


if __name__ == '__main__':
    train()
    export()