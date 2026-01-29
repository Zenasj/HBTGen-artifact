# tf.random.uniform((24, 48, 1024, 3), dtype=tf.float32) ← Input shape: batch=24, height=48, width=1024, channels=3

import tensorflow as tf

layers = tf.keras.layers

class MyModel(tf.keras.Model):
    def __init__(self, configs, name=None):
        super(MyModel, self).__init__(name=name)
        self.vocab_size = configs['vocab_size']

        self.conv1 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')

        self.conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')

        self.conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')
        self.conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')

        self.padd4 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')

        self.conv5 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')
        self.bncv5 = layers.BatchNormalization(axis=-1, name='bnconv5')

        self.conv6 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')
        self.bncv6 = layers.BatchNormalization(axis=-1, name='bnconv6')

        self.pddd6 = layers.ZeroPadding2D(padding=(0, 1))
        self.pool6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')

        self.conv7 = layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='valid', name='conv7')

        self.final_layer = layers.Dense(self.vocab_size, name='ctc_decoder_linear')

    def get_feature_step(self, widths):
        # Feature map width is reduced approximately by factor 4 after conv/pool layers
        return tf.cast(tf.cast(widths, tf.float32) / 4.0, dtype=tf.int32)

    @tf.function
    def call(self, inputs, widths, training=True):
        # inputs shape: [batch, height, width, channels]
        features = self.conv1(inputs)
        features = self.pool1(features)
        features = self.conv2(features)
        features = self.pool2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.padd4(features)
        features = self.pool4(features)
        features = self.conv5(features)
        features = self.bncv5(features, training=training)
        features = self.conv6(features)
        features = self.bncv6(features, training=training)
        features = self.pddd6(features)
        features = self.pool6(features)
        features = self.conv7(features)
        # Reduce height dimension by max pooling (axis=1)
        cnn_features = tf.reduce_max(features, axis=1)
        # Optionally RNN layers can go here, but replaced by cnn_features directly
        rnn_features = cnn_features
        final_logits = self.final_layer(rnn_features)  # shape: [batch, seq_len, vocab_size]
        widths = self.get_feature_step(widths)
        return cnn_features, rnn_features, widths, final_logits


def my_model_function():
    # Provide vocab_size consistent with examples
    configs = {'vocab_size': 1424}
    return MyModel(configs)


def GetInput():
    batch = 24
    imgh = 48
    imgw = 1024
    imgc = 3
    # Generate input tensor matching expected input of MyModel
    images = tf.random.uniform([batch, imgh, imgw, imgc], minval=-1, maxval=1, dtype=tf.float32)
    widths = tf.fill([batch], imgw)
    # Return tuple of inputs as expected by call: (inputs, widths)
    return images, widths

