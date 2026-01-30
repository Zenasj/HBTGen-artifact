import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiSpectralDCTLayer(layers.Layer):
    def __init__(self, name, channel, height, width, frequency, **kwargs):
        super(MultiSpectralDCTLayer, self).__init__(name=name)
        self.channel = channel
        self.height = height
        self.width = width
        self.frequency = frequency
        super(MultiSpectralDCTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, w, h, c = input_shape
        self.mapper_x, self.mapper_y = self.get_freq_indices(self.frequency)
        mapper_x = [temp_x * (h // 7) for temp_x in self.mapper_x]
        mapper_y = [temp_y * (w // 7) for temp_y in self.mapper_y]
        self.dynamic_weight = tf.constant(
            self.get_dct_filter(self.height, self.width, mapper_x, mapper_y, channel=self.channel))
        super().build(input_shape)

    def get_config(self):
        base_config = super(MultiSpectralDCTLayer, self).get_config()
        config = {
            'channel': self.channel,
            'height': self.height,
            "width": self.width,
            "frequency": self.frequency,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def get_freq_indices(self, methods):
        assert methods in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                           'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                           'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
        num_freq = int(methods[3:])
        if 'top' in methods:
            all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2,
                                 2,
                                 6, 1]
            all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3,
                                 0,
                                 5, 3]
            mapper_x = all_top_indices_x[:num_freq]
            mapper_y = all_top_indices_y[:num_freq]
        elif 'low' in methods:
            all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1,
                                 2,
                                 3, 4]
            all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6,
                                 5,
                                 4, 3]

            mapper_x = all_low_indices_x[:num_freq]
            mapper_y = all_low_indices_y[:num_freq]

        elif 'bot' in methods:
            all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5,
                                 5,
                                 3, 6]
            all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5,
                                 3,
                                 3, 3]
            mapper_x = all_bot_indices_x[:num_freq]
            mapper_y = all_bot_indices_y[:num_freq]
        else:
            raise NotImplementedError
        return mapper_x, mapper_y

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_numpy_filter = np.zeros(shape=(tile_size_y, tile_size_x, channel), dtype=np.float32)
        c_part = channel // len(mapper_x)

        for i, (u_x, u_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_numpy_filter[t_y, t_x, i * c_part: (i + 1) * c_part] = self.build_filter(t_x, u_x,
                                                                                                 tile_size_x) * self.build_filter(
                        t_y, u_y, tile_size_y)
        return dct_numpy_filter

    def build_filter(self, pos, frequency, POS):
        result = np.cos(np.pi * frequency * (pos + 0.5) / POS) / np.sqrt(POS)
        if frequency == 0:
            return result
        else:
            return result * np.sqrt(2)

    def call(self, inputs, training=None):
        assert len(inputs.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(inputs.shape))
        x = inputs * self.dynamic_weight
        result = tf.math.reduce_sum(x, axis=[1, 2], keepdims=True)
        return result


class MultiSpectralAttentionLayer(layers.Layer):
    def __init__(self, name, reduction=16, freq_sel_method='top16', **kwargs):
        super(MultiSpectralAttentionLayer, self).__init__(name=name)
        self.reduction = int(reduction)
        self.freq_sel_method = freq_sel_method
        super(MultiSpectralAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        c2wh = dict([(256, 112), (512, 56), (1024, 28), (2048, 14)])
        _, h, w, c = input_shape

        self.dct_h = c2wh[c]
        self.dct_w = c2wh[c]
        self.channel = c

        self.dct_layer = MultiSpectralDCTLayer("DCT", self.channel, self.dct_h, self.dct_w, self.freq_sel_method)
        self.fc1 = layers.Dense(self.channel // self.reduction, use_bias=True, activation='relu')
        self.fc2 = layers.Dense(self.channel, use_bias=True, activation='sigmoid')
        self.reshapeTensor = tf.keras.layers.Reshape((1, 1, c), name=f'Reshape_None_1_1_{c}')

        super().build(input_shape)

    def get_config(self):
        base_config = super(MultiSpectralAttentionLayer, self).get_config()
        config = {
            "reduction": self.reduction,
            "freq_sel_method": self.freq_sel_method,
        }
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def call(self, inputs, training=None):
        x = self.dct_layer(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshapeTensor(x)
        return inputs * x