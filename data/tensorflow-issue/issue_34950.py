from tensorflow import keras
from tensorflow.keras import layers

model.save_weights("./save_weights/resNet_{}.h5".format(epoch))

feature = resnet101(num_classes=5, include_top=False)
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(5)])
model.load_weights('./save_weights/resNet_5.h5', by_name=True)

import tensorflow as tf


def my_model(im_height=64, im_width=64):
    in_im = tf.keras.Input(shape=(im_height, im_width, 3))
    # 64x64x3
    x = tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=2, padding="same", use_bias=False)(in_im)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # 32x32x32
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    # 16x16x32
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # 8x8x64

    model = tf.keras.Model(inputs=in_im, outputs=x)

    return model


# save weights
feature = my_model()
feature.trainable = False
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(5)])
model.save_weights('my_net.h5')

# load weights
feature = my_model()
# feature.trainable = False
model = tf.keras.Sequential([feature,
                             tf.keras.layers.GlobalAvgPool2D(),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(1024),
                             tf.keras.layers.Dropout(rate=0.2),
                             tf.keras.layers.Dense(5)])
model.load_weights('my_net.h5')