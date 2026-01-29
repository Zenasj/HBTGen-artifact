# tf.random.uniform((B, H, W, C), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class MultiSpectralDCTLayer(layers.Layer):
    def __init__(self, name, channel, height, width, frequency, **kwargs):
        super(MultiSpectralDCTLayer, self).__init__(name=name, **kwargs)
        self.channel = channel
        self.height = height
        self.width = width
        self.frequency = frequency

    def build(self, input_shape):
        # input_shape expected as (batch, height, width, channels)
        _, w, h, c = input_shape
        self.mapper_x, self.mapper_y = self.get_freq_indices(self.frequency)

        # The original code multiplies mapper points by (h // 7) and (w // 7) respectively.
        # It seems h and w might be flipped, (w,h) from input_shape and tile_size params.
        mapper_x = [temp_x * (h // 7) for temp_x in self.mapper_x]
        mapper_y = [temp_y * (w // 7) for temp_y in self.mapper_y]

        # Generate the static DCT filter weights as a numpy array, then convert to tf.constant.
        # This will be broadcast-multiplied in call().
        dct_filter = self.get_dct_filter(self.height, self.width, mapper_x, mapper_y, channel=self.channel)
        self.dynamic_weight = tf.constant(dct_filter, dtype=tf.float32)

        super(MultiSpectralDCTLayer, self).build(input_shape)

    def get_config(self):
        base_config = super(MultiSpectralDCTLayer, self).get_config()
        config = {
            'channel': self.channel,
            'height': self.height,
            'width': self.width,
            'frequency': self.frequency,
        }
        return dict(list(base_config.items()) + list(config.items()))

    def get_freq_indices(self, method):
        # Validate and parse frequency selection method strings like 'top16', 'bot8', etc.
        assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                          'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                          'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
        num_freq = int(method[3:])
        if 'top' in method:
            all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2,
                                 2, 6, 1]
            all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3,
                                 0, 5, 3]
            mapper_x = all_top_indices_x[:num_freq]
            mapper_y = all_top_indices_y[:num_freq]
        elif 'low' in method:
            all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1,
                                 2, 3, 4]
            all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6,
                                 5, 4, 3]

            mapper_x = all_low_indices_x[:num_freq]
            mapper_y = all_low_indices_y[:num_freq]

        elif 'bot' in method:
            all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5,
                                 5, 3, 6]
            all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5,
                                 3, 3, 3]
            mapper_x = all_bot_indices_x[:num_freq]
            mapper_y = all_bot_indices_y[:num_freq]
        else:
            raise NotImplementedError("Frequency selection method not supported.")
        return mapper_x, mapper_y

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        # Construct DCT filters using numpy (as in the original) since TF tensors cannot be assigned 
        # in arbitrary slices easily.
        dct_numpy_filter = np.zeros(shape=(tile_size_y, tile_size_x, channel), dtype=np.float32)
        c_part = channel // len(mapper_x)

        for i, (u_x, u_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    # The resulting block in channel dimension is assigned the product of two 1D DCT filter values.
                    val = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, u_y, tile_size_y)
                    dct_numpy_filter[t_y, t_x, i * c_part:(i + 1) * c_part] = val
        return dct_numpy_filter

    def build_filter(self, pos, frequency, POS):
        # Helper to compute DCT filter coefficient as in original code.
        result = np.cos(np.pi * frequency * (pos + 0.5) / POS) / np.sqrt(POS)
        if frequency == 0:
            return result
        else:
            return result * np.sqrt(2)

    def call(self, inputs, training=None):
        # inputs shape: (B, H, W, C)
        assert len(inputs.shape) == 4, f'Input tensor must be 4D but got {len(inputs.shape)}D'
        # Multiply input by precomputed DCT weights (broadcasting on batch dim)
        x = inputs * self.dynamic_weight
        # Summing spatially to aggregate spectral coefficients
        result = tf.math.reduce_sum(x, axis=[1, 2], keepdims=True)
        return result


class MultiSpectralAttentionLayer(layers.Layer):
    def __init__(self, name, reduction=16, freq_sel_method='top16', **kwargs):
        super(MultiSpectralAttentionLayer, self).__init__(name=name, **kwargs)
        self.reduction = int(reduction)
        self.freq_sel_method = freq_sel_method

    def build(self, input_shape):
        # c2wh maps channel number to DCT block size (height & width) expected
        c2wh = {256: 112, 512: 56, 1024: 28, 2048: 14}
        _, h, w, c = input_shape

        self.dct_h = c2wh.get(c, 14)  # default 14 if channel not found
        self.dct_w = self.dct_h       # DCT window assumed square

        self.channel = c

        # Initialize DCT layer with computed params
        self.dct_layer = MultiSpectralDCTLayer("DCT", self.channel, self.dct_h, self.dct_w, self.freq_sel_method)

        # Two fully connected layers for attention with relu then sigmoid activations
        self.fc1 = layers.Dense(self.channel // self.reduction, use_bias=True, activation='relu')
        self.fc2 = layers.Dense(self.channel, use_bias=True, activation='sigmoid')

        # Reshape output to (batch, 1, 1, channel) to apply channel-wise weights
        self.reshapeTensor = tf.keras.layers.Reshape((1, 1, c))

        super(MultiSpectralAttentionLayer, self).build(input_shape)

    def get_config(self):
        base_config = super(MultiSpectralAttentionLayer, self).get_config()
        config = {
            "reduction": self.reduction,
            "freq_sel_method": self.freq_sel_method,
        }
        return dict(list(base_config.items()) + list(config.items()))

    @tf.function
    def call(self, inputs, training=None):
        x = self.dct_layer(inputs)  # shape: (B,1,1,C)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.reshapeTensor(x)
        # Apply learned channel-wise attention weights
        return inputs * x


class MyModel(tf.keras.Model):
    def __init__(self, *, input_channels=256, reduction=16, freq_sel_method='top16'):
        super(MyModel, self).__init__()
        self.attention = MultiSpectralAttentionLayer(
            name="MultiSpectralAttention",
            reduction=reduction,
            freq_sel_method=freq_sel_method
        )

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        # Forward pass applies multispectral attention
        return self.attention(inputs)


def my_model_function():
    # Example default input_channels assumed 256
    # The model expects 4D tensor inputs of shape (B, H, W, C),
    # where C is input_channels matching MultiSpectralAttentionLayer
    return MyModel(input_channels=256, reduction=16, freq_sel_method='top16')


def GetInput():
    # Assumptions based on the MultiSpectralAttentionLayer expectations:
    # Batch size = 2, Height=56, Width=56, Channels=256
    # Height and width chosen as 56 because c2wh mapping uses 512->56,
    # 256 channels maps to 112 height/width, but model may use 56 as common intermediate value.
    # Since the DCT layer indicates (height,width) passed may differ from input shape,
    # the input tensor shape for demo is chosen to be compatible with channel 256.
    batch_size = 2
    height = 56
    width = 56
    channels = 256
    return tf.random.uniform(shape=(batch_size, height, width, channels), dtype=tf.float32)

