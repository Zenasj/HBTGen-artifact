import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

loss_tracker = tf.keras.metrics.Mean(name="loss")


class TestModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
        self.Dense = tf.keras.layers.Dense(80)
        self.MSE = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def train_step(self, data):
        batch_size = tf.squeeze(tf.slice(tf.shape(data), [0], [1]), -1)
        #tried hardcoding batch_size to e.g. 16 with two workers -> does not resolve the issue
        max_length = 10
        i_start = tf.constant(0, dtype=tf.int32)
        dummy_inputs = tf.zeros([batch_size, 1, 200])
        gen_outputs = tf.zeros([batch_size, 0, 80], tf.float32)

        def body(i, input, output_full):
            output_single = self.Dense(input)
            output_full = tf.concat([output_full, output_single], 1)
            i_next = i + 1
            return i_next, input, output_full

        with tf.GradientTape() as tape:
            _, _, gen_data = tf.while_loop(cond=lambda i,
                                                       input,
                                                       output_full: tf.less(i, max_length),
                                           body=body,
                                           loop_vars=(i_start,
                                                      dummy_inputs,
                                                      gen_outputs,),
                                           shape_invariants=(i_start.get_shape(),
                                                             tf.TensorShape([None, None, 200]),
                                                             tf.TensorShape([None, None, 80])))
            loss = self.MSE(data, gen_data)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}


def get_dummy_data():
    x = np.random.random((128, 10, 80))
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset.options().experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.batch(32)
    return dataset


def get_model():
    model = TestModel()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    return model

import os
import json
import tensorflow as tf
os.environ.pop('TF_CONFIG', None)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.RING)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

import TestModel

with strategy.scope():
    model = TestModel.get_model()

data = TestModel.get_dummy_data()
model.fit(data, epochs=4)

import os
import json
import tensorflow as tf
os.environ.pop('TF_CONFIG', None)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 1}
}
os.environ['TF_CONFIG'] = json.dumps(tf_config)

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.RING)
strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

import TestModel

with strategy.scope():
    model = TestModel.get_model()

data = TestModel.get_dummy_data()
model.fit(data, epochs=4)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))