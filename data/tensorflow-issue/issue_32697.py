# tf.random.uniform((16, 256, 256, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the UNet with shrinkFactor=4 and input size (256,256,1)
        shrinkFactor = 4
        inputs = tf.keras.layers.Input(shape=(256, 256, 1))

        # Encoder
        conv1 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(256//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(0.0)(conv4)
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop4)

        conv5 = tf.keras.layers.Conv2D(1024//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = tf.keras.layers.Conv2D(1024//shrinkFactor, 3, activation='relu', padding='same',
                                       kernel_initializer='he_normal', name='code_')(conv5)
        drop5 = tf.keras.layers.Dropout(0.0)(conv5)

        # Decoder
        up6 = tf.keras.layers.Conv2D(512//shrinkFactor, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(drop5))
        merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
        conv6 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = tf.keras.layers.Conv2D(512//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = tf.keras.layers.Conv2D(192//shrinkFactor, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv6))
        merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
        conv7 = tf.keras.layers.Conv2D(192//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = tf.keras.layers.Conv2D(192//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = tf.keras.layers.Conv2D(128//shrinkFactor, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv7))
        merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
        conv8 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = tf.keras.layers.Conv2D(128//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = tf.keras.layers.Conv2D(64//shrinkFactor, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(conv8))
        merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
        conv9 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(64//shrinkFactor, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv9 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = tf.keras.layers.Conv2D(1, 1, activation='relu')(conv9)

        # Normalize output to [0,1]
        r = tf.keras.layers.Lambda(lambda x: (x - tf.keras.backend.min(x)) / (tf.keras.backend.max(x) - tf.keras.backend.min(x)),
                                   name='reconstruction')(conv10)

        self.model = tf.keras.Model(inputs=inputs, outputs=r)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        return self.model(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random float32 tensor input of shape (16,256,256,1) matching batch size and image dimensions
    return tf.random.uniform((16, 256, 256, 1), dtype=tf.float32)

