# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32)  ← Assuming input shape is CIFAR-10 images (batch, 32, 32, 3)

import tensorflow as tf
from tensorflow.keras import layers

class Kwta(layers.Layer):
    def __init__(self, ratio, conv=False, data_format="channels_last", **kwargs):
        super(Kwta, self).__init__(**kwargs)
        self.conv = conv
        # Use tf.Variable to hold ratio so that training-time updates are saved and restored properly
        self.ratio = tf.Variable(ratio, trainable=False, dtype=tf.float32)
        self.data_format = data_format
        self.dim = 1 if self.data_format == "channels_last" else 2

    def build(self, input_shape):
        # For conv layers, dimension we consider is channels axis
        # self.dim should indicate the channel axis index
        # No trainable weights needed here
        super(Kwta, self).build(input_shape)

    def call(self, inputs):
        # The original poster's code uses ratio internally in some custom computation.
        # Since only integers can be used inside call for computation (per issue),
        # convert ratio float to int threshold on channels dimension and perform top-k selection.
        # "k-winners-take-all" activation means selecting top-k elements/channel-wise.

        # For conv=True, inputs shape is (B, H, W, C) with channels_last
        if self.conv:
            # Compute k according to ratio and channels
            shape = tf.shape(inputs)
            if self.data_format == "channels_last":
                channels = shape[-1]
                # Convert ratio to int count: how many winners to keep in channels dimension
                k = tf.cast(tf.math.round(self.ratio * tf.cast(channels, tf.float32)), tf.int32)
                k = tf.maximum(k, 1)
                # For each spatial location, zero out all but top k values across channels
                # We'll do this by taking top-k across channels dimension

                # Reshape input to (B*H*W, C)
                flat_inputs = tf.reshape(inputs, [-1, channels])
                # Find top-k values and min threshold
                top_k_values, _ = tf.math.top_k(flat_inputs, k=k, sorted=True)
                threshold = tf.expand_dims(top_k_values[:, -1], axis=1)
                # Zero out values below threshold
                mask = tf.cast(flat_inputs >= threshold, inputs.dtype)
                outputs = flat_inputs * mask
                # Reshape back
                outputs = tf.reshape(outputs, shape)
            else:
                # channels_first
                channels = shape[self.dim]
                k = tf.cast(tf.math.round(self.ratio * tf.cast(channels, tf.float32)), tf.int32)
                k = tf.maximum(k, 1)
                # inputs shape: (B, C, H, W)
                # Reshape to (B, C, H*W)
                flat_inputs = tf.reshape(inputs, [shape[0], channels, -1])
                top_k_values, _ = tf.math.top_k(flat_inputs, k=k, sorted=True)
                threshold = tf.expand_dims(top_k_values[:, :, -1], axis=2)
                mask = tf.cast(flat_inputs >= threshold, inputs.dtype)
                outputs = flat_inputs * mask
                # Reshape back to original
                outputs = tf.reshape(outputs, shape)
            return outputs
        else:
            # Non convolutional mode, assume inputs shape (B, features)
            # Similar k-winners take all in features dimension
            shape = tf.shape(inputs)
            features = shape[-1]
            k = tf.cast(tf.math.round(self.ratio * tf.cast(features, tf.float32)), tf.int32)
            k = tf.maximum(k, 1)
            top_k_values, _ = tf.math.top_k(inputs, k=k, sorted=True)
            threshold = tf.expand_dims(top_k_values[:, -1], axis=1)
            mask = tf.cast(inputs >= threshold, inputs.dtype)
            outputs = inputs * mask
            return outputs

    def get_config(self):
        # Properly save the ratio value as a float, extracting from tf.Variable
        config = {
            'conv': self.conv,
            'ratio': float(self.ratio.numpy()),
            'data_format': self.data_format,
            'dim': self.dim
        }
        base_config = super(Kwta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.seq_model = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation=None, input_shape=(32, 32, 3)),
            Kwta(ratio=0.6, conv=True, name='kwta_1'),
            layers.Conv2D(32, 3, padding='same', activation=None),
            Kwta(ratio=0.6, conv=True, name='kwta_2'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2, seed=42),

            layers.Conv2D(64, 3, padding='same', activation=None),
            Kwta(ratio=0.6, conv=True, name='kwta_3'),
            layers.Conv2D(64, 3, padding='same', activation=None),
            Kwta(ratio=0.6, conv=True, name='kwta_4'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3, seed=42),

            layers.Flatten(),
            layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        ])

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        return self.seq_model(inputs, training=training)


def my_model_function():
    # Return an instance of MyModel with default initialization.
    return MyModel()


def GetInput():
    # Returns a random input tensor matching CIFAR-10 image shape (32,32,3)
    # Assuming batch size 1 for demonstration
    return tf.random.uniform((1, 32, 32, 3), dtype=tf.float32)

