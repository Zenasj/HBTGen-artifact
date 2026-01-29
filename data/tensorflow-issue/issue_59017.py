# tf.random.uniform((1, 64, 64, 4), dtype=tf.float32) ← Assumed input shape (batch=1, height=64, width=64, channels=4)
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D((padding, padding))
        self.conv2d = keras.layers.Conv2D(
            channels, kernel_size, strides=(stride, stride), padding='valid'
        )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)


class ResnetBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv1 = PaddedConv2D(out_channels, 3, padding=1)
        self.norm2 = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.conv2 = PaddedConv2D(out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = PaddedConv2D(out_channels, 1)
        else:
            self.nin_shortcut = lambda x: x

    def call(self, x):
        h = self.conv1(tf.keras.activations.swish(self.norm1(x)))
        h = self.conv2(tf.keras.activations.swish(self.norm2(h)))
        return self.nin_shortcut(x) + h


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(channels, 1)
        self.k = PaddedConv2D(channels, 1)
        self.v = PaddedConv2D(channels, 1)
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute attention
        b, h, w, c = tf.shape(q)[0], tf.shape(q)[1], tf.shape(q)[2], tf.shape(q)[3]
        # Reshape and transpose to use batch matmul
        q_reshaped = tf.reshape(q, (b, h * w, c))  # b, hw, c
        k_reshaped = tf.transpose(tf.reshape(k, (b, h * w, c)), perm=[0, 2, 1])  # b, c, hw

        w_ = tf.matmul(q_reshaped, k_reshaped)  # b, hw, hw
        scale = tf.math.rsqrt(tf.cast(c, tf.float32))
        w_ = w_ * scale
        w_ = tf.nn.softmax(w_, axis=-1)

        v_reshaped = tf.transpose(tf.reshape(v, (b, h * w, c)), perm=[0, 2, 1])  # b, c, hw
        h_ = tf.matmul(v_reshaped, tf.transpose(w_, perm=[0, 2, 1]))  # b, c, hw
        h_ = tf.transpose(h_, perm=[0, 2, 1])  # b, hw, c
        h_ = tf.reshape(h_, (b, h, w, c))

        return x + self.proj_out(h_)


class Decoder(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                keras.layers.Lambda(lambda x: x / 0.18215),  # scale input
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 512),
                AttentionBlock(512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                ResnetBlock(512, 512),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512, 256),
                ResnetBlock(256, 256),
                ResnetBlock(256, 256),
                keras.layers.UpSampling2D(size=(2, 2)),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(256, 128),
                ResnetBlock(128, 128),
                ResnetBlock(128, 128),
                tfa.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ]
        )


class MyModel(tf.keras.Model):
    def __init__(self, img_height=512, img_width=512, download_weights=True):
        super().__init__()
        self.n_h = img_height // 8  # latent height
        self.n_w = img_width // 8   # latent width
        self.decoder = Decoder()

        # Flag to optionally load weights if available
        if download_weights:
            try:
                decoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
                    file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
                )
                self.decoder.load_weights(decoder_weights_fpath)
            except Exception:
                # Weights may not be downloaded/available; silently pass
                pass

    def call(self, inputs):
        # Ensure inputs have batch dimension for TFLite compatibility
        # If inputs shape is (n_h, n_w, 4), add batch dim 1
        if tf.rank(inputs) == 3:
            inputs = tf.reshape(inputs, (1, self.n_h, self.n_w, 4))
        return self.decoder(inputs)


def my_model_function():
    # Return an instance of MyModel with default input size 512x512 latent space assumed
    return MyModel()


def GetInput():
    # Generate a random input tensor matching the expected latent shape (batch=1 assumed)
    batch = 1
    n_h = 512 // 8  # 64
    n_w = 512 // 8  # 64
    channels = 4
    return tf.random.uniform((batch, n_h, n_w, channels), dtype=tf.float32)

