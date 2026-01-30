import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
tf.debugging.set_log_device_placement(False)

import op_utils
from nets import ResNet

parser = argparse.ArgumentParser(description='')

parser.add_argument("--gpu_id", default= [0], type=int, nargs = '+')
parser.add_argument("--compile", default=True, action = 'store_true')
parser.add_argument("--learning_rate", default=1e-1, type=float)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--batch_size", default=128, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[i] for i in args.gpu_id], 'GPU')
    for gpu_id in args.gpu_id:
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
    devices = ['/gpu:{}'.format(i) for i in args.gpu_id]
    strategy = tf.distribute.MirroredStrategy(devices, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    with strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        def inference(image, label):
            image = tf.cast(image, tf.float32)
            image = (image-np.array([113.9,123.0,125.3]))/np.array([66.7,62.1,63.0])
            return image, label

        from tensorflow.keras.datasets.cifar100 import load_data
        (train_images, train_labels), (test_images, test_labels) = load_data()

        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_ds = test_ds.map(inference, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.batch(args.batch_size)
        test_ds = test_ds.with_options(options)
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

        model = ResNet.Model(num_layers = 56, num_class = 100, name = 'ResNet', trainable = True)

        train_step, train_loss, train_accuracy, optimizer = op_utils.Optimizer(args, model, strategy)

        for step, data in enumerate(test_ds):
            train_step(*data)
            print('check')