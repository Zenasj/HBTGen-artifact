from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence


def get_model(input_shape, output_shape, compile_batch_size):
    assert K.image_data_format() == 'channels_last'
    m_input = Input(shape=input_shape, batch_size=compile_batch_size)
    m_output = Conv2D(output_shape[-1], 1, activation=None)(m_input)
    model = Model(inputs=m_input, outputs=m_output)
    return model


class DataGenerator(Sequence):
    def __init__(self, batch_size, image_size, num_batches):
        assert K.image_data_format() == 'channels_last'

        self.batch_size = batch_size
        self.image_size = image_size
        self.num_batches = num_batches
        self.on_epoch_end()

    def __len__(self):
        assert self.num_batches > 0
        return self.num_batches

    def __getitem__(self, index):
        return np.zeros((self.batch_size,) + self.image_size).astype('float32'), np.zeros(
            (self.batch_size,) + self.image_size).astype('float32')

    def on_epoch_end(self):
        pass


if __name__ == '__main__':
    print(tf.version.GIT_VERSION, tf.version.VERSION)

    # set to True to trigger infinite wait
    if True:
        # limit GPU memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        tf.keras.backend.set_session(tf.Session(config=config))

    inp_shape = (128, 64, 2)
    outp_shape = (128, 64, 2)
    batch_size = 16

    gen = DataGenerator(batch_size, inp_shape, 10)
    model = get_model(inp_shape, outp_shape, batch_size)
    model.summary()
    model.compile(loss=["mse"], optimizer="adam", metrics=["accuracy"])
    model.fit_generator(gen, steps_per_epoch=10, epochs=5, workers=2, use_multiprocessing=True)