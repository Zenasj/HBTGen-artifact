# tf.random.uniform((4, 512, 512, 1), dtype=tf.float32) ← Based on batch_size=4 and input_size=(512, 512, 1) in unet model

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import SGD


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the U-Net model architecture manually inside the subclass

        input_size = (512, 512, 1)

        self.inputs_layer = Input(input_size)

        # Encoder path
        self.conv1_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv1_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2_1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2_2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3_1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3_2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv4_1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4_2 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop4 = layers.Dropout(0.5)
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv5_1 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5_2 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.drop5 = layers.Dropout(0.5)

        # Decoder path
        # Corrected parentheses around kernel_initializer argument in up6 definition
        self.up6_upsample = layers.UpSampling2D(size=(2, 2))
        self.up6_conv = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6_1 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6_2 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up7_upsample = layers.UpSampling2D(size=(2, 2))
        self.up7_conv = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_1 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7_2 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up8_upsample = layers.UpSampling2D(size=(2, 2))
        self.up8_conv = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_1 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8_2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up9_upsample = layers.UpSampling2D(size=(2, 2))
        self.up9_conv = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_2 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9_3 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.conv10 = layers.Conv2D(1, 1, activation='sigmoid')

        # Because we're subclassing tf.keras.Model directly,
        # set up the optimizer and loss here (could be done outside too)
        self.optimizer = SGD(learning_rate=0.015)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()

    def call(self, inputs, training=False):
        # Encoder
        conv1 = self.conv1_1(inputs)
        conv1 = self.conv1_2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        drop4 = self.drop4(conv4, training=training)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)
        drop5 = self.drop5(conv5, training=training)

        # Decoder
        up6 = self.up6_upsample(drop5)
        up6 = self.up6_conv(up6)
        merge6 = tf.concat([drop4, up6], axis=3)
        conv6 = self.conv6_1(merge6)
        conv6 = self.conv6_2(conv6)

        up7 = self.up7_upsample(conv6)
        up7 = self.up7_conv(up7)
        merge7 = tf.concat([conv3, up7], axis=3)
        conv7 = self.conv7_1(merge7)
        conv7 = self.conv7_2(conv7)

        up8 = self.up8_upsample(conv7)
        up8 = self.up8_conv(up8)
        merge8 = tf.concat([conv2, up8], axis=3)
        conv8 = self.conv8_1(merge8)
        conv8 = self.conv8_2(conv8)

        up9 = self.up9_upsample(conv8)
        up9 = self.up9_conv(up9)
        merge9 = tf.concat([conv1, up9], axis=3)
        conv9 = self.conv9_1(merge9)
        conv9 = self.conv9_2(conv9)
        conv9 = self.conv9_3(conv9)

        conv10 = self.conv10(conv9)

        return conv10

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        # Override compile to set defaults matching original unet
        if optimizer is None:
            optimizer = self.optimizer
        if loss is None:
            loss = self.loss_fn
        if metrics is None:
            metrics = [self.accuracy_metric]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)


def my_model_function():
    # Return an instance of MyModel, with default weights initialized.
    model = MyModel()
    # Compile to match original setup
    model.compile()
    return model


def GetInput():
    # Return a random tensor input matching the model input shape
    # From original code: batch_size=4, input_size=(512, 512, 1)
    batch_size = 4
    input_shape = (batch_size, 512, 512, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

