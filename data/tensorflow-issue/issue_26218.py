# tf.random.uniform((86, 128, 128, 3), dtype=tf.float32)  # batch_size=86, input shape (128,128,3)

import tensorflow as tf

KL = tf.keras.layers
K = tf.keras.backend

class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4; Received input shape: {}'.format(input_shape))

        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4; Received input shape: {}'.format(input_shape))

        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
        channels = input_shape[3] // (self.size[0] * self.size[1])

        if channels * self.size[0] * self.size[1] != input_shape[3]:
            raise ValueError('channels of input and size are incompatible')

        return (input_shape[0], height, width, channels)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(PixelShuffler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def upscale(dim):
    # Wrapper layer to apply PixelShuffler after Conv2D with dim*4 filters
    def func(x):
        x = KL.Conv2D(dim * 4, kernel_size=3, strides=1, padding='same')(x)
        return PixelShuffler()(x)
    return func

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers explicitly for compatibility with TF 2.20.0 XLA compilation

        self.conv1 = KL.Conv2D(128, 5, strides=2, padding='same')
        self.conv2 = KL.Conv2D(256, 5, strides=2, padding='same')
        self.conv3 = KL.Conv2D(512, 5, strides=2, padding='same')
        self.conv4 = KL.Conv2D(1024, 5, strides=2, padding='same')
        self.flatten = KL.Flatten()
        self.dense1 = KL.Dense(1024)
        self.dense2 = KL.Dense(8 * 8 * 1024)
        self.reshape = KL.Reshape((8, 8, 1024))

        # Upscale pixel shuffler blocks with specified dims
        # Note: Each upscale block includes Conv2D(dim*4) followed by PixelShuffler
        self.upscale1_conv = KL.Conv2D(512 * 4, kernel_size=3, strides=1, padding='same')
        self.upscale1_ps = PixelShuffler()
        self.upscale2_conv = KL.Conv2D(256 * 4, kernel_size=3, strides=1, padding='same')
        self.upscale2_ps = PixelShuffler()
        self.upscale3_conv = KL.Conv2D(128 * 4, kernel_size=3, strides=1, padding='same')
        self.upscale3_ps = PixelShuffler()
        self.upscale4_conv = KL.Conv2D(64 * 4, kernel_size=3, strides=1, padding='same')
        self.upscale4_ps = PixelShuffler()

        self.final_conv = KL.Conv2D(3, 5, strides=1, padding='same')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)

        x = self.upscale1_conv(x)
        x = self.upscale1_ps(x)
        x = self.upscale2_conv(x)
        x = self.upscale2_ps(x)
        x = self.upscale3_conv(x)
        x = self.upscale3_ps(x)
        x = self.upscale4_conv(x)
        x = self.upscale4_ps(x)

        x = self.final_conv(x)
        return x

def my_model_function():
    # Create and compile the model with the optimizer and loss as in original code
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, beta_1=0.5, beta_2=0.999),
        loss='mae'
    )
    return model

def GetInput():
    # Generate a random tensor matching the input expected by MyModel: shape (86, 128, 128, 3)
    # dtype=float32 by default
    return tf.random.uniform((86, 128, 128, 3), dtype=tf.float32)

