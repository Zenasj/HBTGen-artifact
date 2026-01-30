import random
from tensorflow import keras
from tensorflow.keras import layers

import json
import subprocess

import os


def create_TF_config(name, id):
    return {
        'cluster': {
            'worker': ['localhost:9999']
            ,'chief': ['localhost:9997']
        },
        'task': {'type': name, 'index': id}
    }


def set_TF_CONFIG(id, name='worker'):
    os.environ['TF_CONFIG'] = json.dumps(create_TF_config(name, id))


def start_processes(cluster_def, key, device=None):
    process_list = []
    if key in cluster_def:
        for i, _ in enumerate(cluster_def[key]):
            if device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
                device +=1
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

            process_list.append(subprocess.Popen(['python', 'distributed_training_minimal_example.py', '--job-name='+key, '--job-id=' + str(i)]))
    return process_list, device


if __name__ == "__main__":
    cluster_def = create_TF_config("","")['cluster']

    process_list = []
    #this_list, device = start_processes(cluster_def, 'chief')
    device=0
    for key in ['chief','worker', 'ps']:
        this_list, device = start_processes(cluster_def, key, device)
        process_list.extend(this_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['TF_CONFIG'] = "{}"

    for p in process_list:
        p.wait()

from run_distributed_training_minimal_example import create_TF_config, set_TF_CONFIG

use_custom_check_point = False

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-name',
                        type=str,
                        default="worker",
                        help='type of job this process is running')
    parser.add_argument('--job-id',
                        type=int,
                        default=0,
                        help='id of this job type for this process to run')
    return parser.parse_args()

args = parse_arguments()

tf_config = create_TF_config("", "")
cluster_def = tf_config['cluster']
set_TF_CONFIG(args.job_id, args.job_name)

is_chief = args.job_name == 'chief'
print('is_chief:'+str(is_chief))
batchSize = len(cluster_def['worker'])

import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():

    def create_simple_model():
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.04), input_shape = (128, 128, 1)),
            tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same',kernel_regularizer = tf.keras.regularizers.l2(0.04)),
            tf.keras.layers.Dense(1, activation='softmax')
        ])


    def localised_cross_entropy(y_true, y_pred, ratio=1.0):
        positive_error = ratio * y_true * tf.keras.backend.log(0.0000001 + y_pred)
        negative_error = (1 - y_true) * tf.keras.backend.log(1.0000001 - y_pred)
        errors = positive_error + negative_error
        return tf.keras.backend.mean(errors)

    def localised_cross_entropy_loss(y_true, y_pred, ratio=1.0):
        return -localised_cross_entropy(y_true, y_pred, ratio)

    def create_data():
        import numpy as np
        data_set = []
        for i in range(20):
            ip = np.random.random([128, 128, 1]).astype(np.float32)
            op = np.random.randint(0, 2, [128, 128, 1]).astype(np.float32)
            data_set.append((ip, op))
        return data_set

    model = create_simple_model()
    model.summary()

    trainingData = create_data()
    model.compile('adam', loss=localised_cross_entropy_loss, metrics=[localised_cross_entropy])

    split = int(len(trainingData)*0.8)
    trainData, valData = trainingData[:split], trainingData[split:]

    def create_RAM_generator(data):
        while True:
            for i in data:
                yield i

    def tensorflow_generator_training(data_getter, batchSize=None):
        import tensorflow as tf

        def __getter_generator():
            while True:
                item = next(data_getter)
                yield item

        shapes = ((None, None, 1), (None, None, 1))
        dataset = tf.data.Dataset.from_generator(generator=__getter_generator, output_types=(tf.float32, tf.float32),output_shapes=shapes)
        if batchSize is not None:
            dataset = dataset.batch(batchSize)
        return dataset

    def generatorise(data):
        train_gen = create_RAM_generator(data)
        train_gen = tensorflow_generator_training(train_gen, batchSize=batchSize)
        return train_gen

    train_gen = generatorise(trainData)
    val_gen = generatorise(valData)

if not use_custom_check_point:
    callbacks_list = [tf.keras.callbacks.ModelCheckpoint('tmp.hdf5')]
else:
    from tensorflow.keras.callbacks import Callback


    class CustomModelCheckpointCallback(Callback):
        def __init__(self, path, model, is_chief_task):
            super(CustomModelCheckpointCallback, self).__init__()
            self.model = model
            self.path = path
            self.is_chief = is_chief_task

        def on_epoch_end(self, epoch, logs=None):
            if self.is_chief:
                self.model.save(self.path)
    callbacks_list = [CustomModelCheckpointCallback('tmp.hdf5', model, is_chief)]

model.fit(train_gen, epochs=3, shuffle=False, callbacks=callbacks_list, validation_data=val_gen, steps_per_epoch=len(trainData), validation_steps=len(valData))