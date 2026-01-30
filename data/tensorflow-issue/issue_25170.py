import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from scipy import signal


DATA_SHAPE = (50, 100, 1)  # single channel image


def prepare_simple_resnet(n_filters=10):

    parameters_input = tf.keras.layers.Input(shape=DATA_SHAPE)

    x = parameters_input
    res_node = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    res_node = tf.keras.layers.ReLU()(res_node)
    res_node = tf.keras.layers.BatchNormalization()(res_node)
    res_node = tf.keras.layers.Conv2D(n_filters, kernel_size=(3, 3), padding='same', use_bias=False)(res_node)
    res_node = tf.keras.layers.ReLU()(res_node)
    res_node = tf.keras.layers.BatchNormalization()(res_node)

    x = tf.keras.layers.add([res_node, x])

    x = tf.keras.layers.Lambda(lambda z: tf.math.reduce_mean(z, axis=[1, 2]))(x)   # mean for each filter
    x = tf.keras.layers.Dense(2)(x)  # output two classes

    model = tf.keras.models.Model(inputs=parameters_input, outputs=x)

    return model


class DataGenerator(tf.keras.utils.Sequence):
    """ Generates random images, which may or may not be blurred"""

    def __init__(self, batch_size=32, random_state=None):
        self.batch_size = batch_size
        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state

        self.averaging_filter = np.ones((3, 3))

    def __len__(self):  # returns number of batches per epoch, not dataset size
        return 50

    def __getitem__(self, index):  # returns batch of data
        batch = []
        labels = []
        for i in range(self.batch_size):
            single_data, label = self.generate_single_data_sample()
            batch.append(single_data)
            labels.append(label)
        return np.array(batch), np.array(labels)

    def generate_single_data_sample(self):
        is_blurred = self.random_state.randint(0, 2)
        data_sample = self.random_state.rand(*DATA_SHAPE[0:2])
        if is_blurred:
            data_sample = signal.convolve2d(data_sample, self.averaging_filter, mode='same')
            label = np.array([0, 1])
        else:
            label = np.array([1, 0])
        data_sample = np.expand_dims(data_sample, -1).astype(np.float32)

        return data_sample, label


if __name__ == '__main__':
    checkpoint_path = '/tmp/test_checkpoint.ckpt'
    tflite_file_path = '/tmp/tflite_model.tflite'
    random_state = np.random.RandomState(seed=0)

    with tf.device("/gpu:0"):

        ############################################################################################
        # LINE BELOW IS IMPORTANT (FOR SOME REASON)
        #  - if set to:
        #        tf.keras.backend.set_learning_phase(0)
        #    model of course is not training, but Lite model returns almost same results as tf.keras model
        #  - if set to:
        #        tf.keras.backend.set_learning_phase(1)
        #    model is training, good acc, but lite model returns completely different results
        #  - if commented/removed
        #    model is training, but there is error in converting model to Lite version
        tf.keras.backend.set_learning_phase(1)
        ############################################################################################

        # train tf.keras model
        data_generator = DataGenerator(random_state=random_state)
        tfkeras_resnet = prepare_simple_resnet()

        tfkeras_resnet.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.00001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False,
                                                         save_best_only=True, verbose=1)
        tfkeras_resnet.fit_generator(generator=data_generator, validation_data=data_generator,
                                     use_multiprocessing=False, epochs=5, callbacks=[cp_callback])

        # test some random data before and after changing learning phase
        random_data = random_state.rand(1, *DATA_SHAPE).astype(np.float32)
        result_tfkeras_train = tfkeras_resnet.predict(random_data)
        tf.keras.backend.set_learning_phase(0)
        result_tfkeras_test = tfkeras_resnet.predict(random_data)

        # Convert to tensorflow lite
        session = tf.keras.backend.get_session()
        converter = tf.lite.TFLiteConverter.from_session(session, [tfkeras_resnet.input],
                                                         [tfkeras_resnet.outputs[0]])
        tflite_model = converter.convert()
        with open(tflite_file_path, 'wb') as f:
            f.write(tflite_model)

        interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(random_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        result_tflite = interpreter.get_tensor(output_details[0]['index'])

        print("TFKeras Train: {}".format(result_tfkeras_train))
        print("TFKeras Test: {}".format(result_tfkeras_test))
        print("TFLite: {}".format(result_tflite))