import random

DEVICE = "/gpu:1"

with tf.device(DEVICE):
    grads = tf.Variable(0.0)
    opt = tf.optimizers.Adam(1e-6)

@tf.function
def train_one_batch(model,train_data,train_label):
    print("tracing batch")
    with tf.device(DEVICE):
        with tf.GradientTape() as tape:
            predicts = model(train_data)
            loss = tf.nn.l2_loss(predicts - train_label)
            grads = tape.gradient(loss, [model.W, model.b])
            opt.apply_gradients(zip(grads, [model.W, model.b]))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pickle

DATA_DIR = "/sensitive_data/"
DEVICE = "/gpu:1"
RANDOM_SEED = 12345
ONEHOT_LENGTH = 1375432
DEBUG = True

if DEBUG:
    tf.debugging.set_log_device_placement(True)
    print(tf.config.experimental.list_physical_devices('GPU')) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]


@tf.function
def _parse_function(example_proto):
    with tf.device("/cpu:0"):
        feature_description = {
            'sensitiveA': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
            'sensitiveB': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
            'sensitiveC': tf.io.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
            'sensitiveD': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=""),
            'sensitiveE': tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.int64, default_value=0, allow_missing=True),
        }
        return tf.io.parse_single_example(example_proto, feature_description)


class Model():
    def __init__(self):
        with tf.device(DEVICE):
            self.W = tf.Variable(tf.random.uniform(shape=[ONEHOT_LENGTH,1],minval=0,maxval=1.0/ONEHOT_LENGTH,seed=RANDOM_SEED))
            self.b = tf.Variable(170.0)

    @tf.function
    def __call__(self, _x):
        with tf.device(DEVICE):
            x = tf.dtypes.cast(_x, tf.float32)
            return tf.matmul(x, self.W) + self.b


with tf.device(DEVICE):
    grads = tf.Variable(0.0)
    opt = tf.optimizers.Adam(1e-6)

@tf.function
def train_one_batch(model,train_data,train_label):
    print("tracing batch")
    with tf.device(DEVICE):
        with tf.GradientTape() as tape:
            predicts = model(train_data)
            loss = tf.nn.l2_loss(predicts - train_label)
            grads = tape.gradient(loss, [model.W, model.b])
            opt.apply_gradients(zip(grads, [model.W, model.b]))   #fixme: causes GPU0 uasge and retrace


if __name__ == '__main__':

    sensitive_tfrecord = pickle.load(open("./sensitive.pkl","rb"))
    for i in range(len(sensitive_tfrecord)):
        sensitive_tfrecord[i] = DATA_DIR + sensitive_tfrecord[i]

    dataset = tf.data.TFRecordDataset(sensitive_tfrecord).map(_parse_function,num_parallel_calls=22)

    with tf.device(DEVICE):
        model = Model()

        for data in dataset.batch(20):
            train_one_batch(model, data["sensitiveE"], data["sensitiveB"])