# tf.random.uniform((None, 540, 540, 3), dtype=tf.float32)  # Input shape inferred from create_model Input layer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom loss is referenced but undefined in the issue.
# Define a simple placeholder custom_loss that does a mean squared error.
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Initial conv layer
        self.conv_init = layers.Conv2D(filters=64, kernel_size=3, strides=1, dilation_rate=2,
                                       padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))

        # Define 8 stages of dense_block + transition_block
        self.dense_blocks1 = []
        self.transition_blocks = []
        for _ in range(8):
            self.dense_blocks1.append(self._make_dense_block(filters=128, layers_=3, bottleneck=True))
            self.transition_blocks.append(self._make_transition_block(filters=128, att=True))

        # Final dense block of 4 layers
        self.dense_block_final = self._make_dense_block(filters=128, layers_=4, bottleneck=True)

        # Final BatchNorm and conv 1x1 to output 45 filters
        self.batchnorm_final = layers.BatchNormalization(axis=-1, fused=True)
        self.conv_final = layers.Conv2D(filters=45, kernel_size=1, strides=1, dilation_rate=1,
                                        padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01))

        # Layers for attention are defined dynamically in attention function

    def call(self, inputs, training=False):
        x = self.conv_init(inputs)
        for dense_block_layer, trans_block in zip(self.dense_blocks1, self.transition_blocks):
            x = dense_block_layer(x, training=training)
            x = trans_block(x, training=training)
        x = self.dense_block_final(x, training=training)
        x = self.batchnorm_final(x, training=training)
        x = self.conv_final(x)
        return x

    def _make_dense_block(self, filters, layers_, bottleneck=True):
        # Returns a function implementing the dense_block logic as a keras layer / callable
        def dense_block_function(x, training=False):
            for _ in range(layers_):
                cb_out = self._conv_block(x, filters, dilation=2, bottleneck=bottleneck, training=training)
                x = tf.keras.layers.concatenate([x, cb_out], axis=-1)
                x = self._attention(x, training=training)
            return x
        return dense_block_function

    def _conv_block(self, x, filters, stride=1, dilation=2, pad='same', bottleneck=True, training=False):
        x_out = layers.BatchNormalization(axis=-1, fused=True)(x, training=training)
        x_out = layers.LeakyReLU(alpha=0.1)(x_out)
        if bottleneck:
            x_out = layers.Conv2D(filters=filters * 4, kernel_size=1, strides=1, dilation_rate=dilation,
                                  padding=pad,
                                  kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x_out)
            x_out = layers.BatchNormalization(axis=-1, fused=True)(x_out, training=training)
            x_out = layers.LeakyReLU(alpha=0.1)(x_out)
        x_out = layers.Conv2D(filters=filters, kernel_size=3, strides=stride, dilation_rate=dilation,
                              padding=pad,
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x_out)
        return x_out

    def _make_transition_block(self, filters, att=True):
        def transition_block_function(x, training=False):
            x = layers.BatchNormalization(axis=-1, fused=True)(x, training=training)
            x = layers.LeakyReLU(alpha=0.1)(x)
            x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, dilation_rate=2,
                              padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)
            x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
            if att:
                x = self._attention(x, training=training)
            return x
        return transition_block_function

    def _attention(self, inputs, training=False):
        x = self._channel_att(inputs, training=training)
        x = self._spatial_att(x, training=training)
        return x

    def _channel_att(self, inputs, ratio=8, training=False):
        channel = inputs.shape[-1]
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        avg_pool = tf.reshape(avg_pool, (-1, 1, 1, channel))
        max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
        max_pool = tf.reshape(max_pool, (-1, 1, 1, channel))

        dense_1 = layers.Dense(channel // ratio, activation='relu')
        dense_2 = layers.Dense(channel, activation='relu')
        avg_out = dense_2(dense_1(avg_pool))
        max_out = dense_2(dense_1(max_pool))
        scale = tf.keras.activations.sigmoid(avg_out + max_out)
        return inputs * scale

    def _spatial_att(self, inputs, kernel=7, training=False):
        avg_pool = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=3)
        conv_layer = layers.Conv2D(filters=1, kernel_size=kernel, padding='same', use_bias=False)
        concat = conv_layer(concat)
        scale = tf.keras.activations.sigmoid(concat)
        return inputs * scale


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile model to mimic original (SGD optimizer, custom loss)
    model.compile(optimizer=keras.optimizers.SGD(), loss=custom_loss)
    return model


def GetInput():
    # Return a random tensor input matching expected input shape (batch size None, 540x540x3)
    # Use batch size 1 for simplicity
    return tf.random.uniform(shape=(1, 540, 540, 3), dtype=tf.float32)

