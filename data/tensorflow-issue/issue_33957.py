from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import math

NUM_CLASSES = 10

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def SEBlock(inputs, input_channels, ratio=0.25):

    num_reduced_filters = max(1, int(input_channels * ratio))
    branch = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # branch = tf.keras.layers.Lambda(lambda branch: tf.expand_dims(input=branch, axis=1))(branch)
    branch = tf.keras.backend.expand_dims(branch, 1)
    branch = tf.keras.backend.expand_dims(branch, 1)
    # branch = tf.keras.layers.Lambda(lambda branch: tf.expand_dims(input=branch, axis=1))(branch)
    branch = tf.keras.layers.Conv2D(filters=num_reduced_filters, kernel_size=(1, 1), strides=1, padding="same")(branch)
    branch = swish(branch)
    branch = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1, 1), strides=1, padding='same')(branch)
    branch = tf.keras.activations.sigmoid(branch)
    output = inputs * branch

    return output

def MBConv(in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, inputs, training=False):
    x = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,kernel_size=(1, 1),strides=1,padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k), strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = SEBlock(x, in_channels*expansion_factor)
    x = swish(x)
    x = tf.keras.layers.Conv2D(filters=out_channels,kernel_size=(1, 1),strides=1,padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    if stride == 1 and in_channels == out_channels:
        if drop_connect_rate:
            x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x, training=training)
        x = tf.keras.layers.Add()([x, inputs])

    return x

def build_mbconv_block(inputs, in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate, training):

    x = inputs
    for i in range(layers):
        if i == 0:
            x = MBConv(in_channels=in_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                       stride=stride, k=k, drop_connect_rate=drop_connect_rate, inputs=x, training=training)
        else:
            x = MBConv(in_channels=out_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                       stride=1, k=k, drop_connect_rate=drop_connect_rate, inputs=x, training=training)

    return x


def EfficientNet(inputs, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2, training=False):

    features = []

    x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),kernel_size=(3, 3),strides=2, padding="same") (inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)

    x = build_mbconv_block(x, in_channels=round_filters(32, width_coefficient),
                           out_channels=round_filters(16, width_coefficient),
                           layers=round_repeats(1, depth_coefficient),
                           stride=1,
                           expansion_factor=1, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(16, width_coefficient),
                           out_channels=round_filters(24, width_coefficient),
                           layers=round_repeats(2, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(24, width_coefficient),
                           out_channels=round_filters(40, width_coefficient),
                           layers=round_repeats(2, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(40, width_coefficient),
                           out_channels=round_filters(80, width_coefficient),
                           layers=round_repeats(3, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(80, width_coefficient),
                           out_channels=round_filters(112, width_coefficient),
                           layers=round_repeats(3, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(112, width_coefficient),
                           out_channels=round_filters(192, width_coefficient),
                           layers=round_repeats(4, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(192, width_coefficient),
                           out_channels=round_filters(320, width_coefficient),
                           layers=round_repeats(1, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient), kernel_size=(1, 1), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x, training=training)
    x = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softmax)(x)

    return x, features


def efficient_net_b0(inputs, training):
    return EfficientNet(inputs,
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        drop_connect_rate=0.2,
                        training=training)

def up_sample(inputs, training=True):
    x = tf.keras.layers.UpSampling2D()(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = tf.keras.layers.ReLU()(x)
    return x

def biggerModel(inputs, outc, training=True):

    _, features =  efficient_net_b0(inputs=inputs, training=training)

    # [ 1/2, 1/4, 1/8, 1/8, 1/16]
    outputs = []
    for i, name in enumerate(features):
        x = features[i]
        if x.shape[1] > inputs.shape[1] // 4:
            continue
        while x.shape[1] < (inputs.shape[1]//4):
            x = up_sample(x, training)
        outputs.append(x)

    quater_res = tf.keras.layers.Concatenate()(outputs)
    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    quater_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='quater', activation=None)(quater_res)

    half_res = up_sample(quater_res, training)
    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(half_res)
    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(half_res)
    half_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='half', activation=None)(half_res)
    
    if training:
        return quater_res_out, half_res_out
    else:
        return quater_res_out

def smallModel(inputs, outc, training=True):


    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(inputs)
    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    quater_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same',  activation=None)(quater_res)


    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(half_res)
    half_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', activation=None)(half_res)

    if training:
        return quater_res_out, half_res_out
    else:
        return quater_res_out

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs = tf.keras.Input(shape=(224, 224, 3), name='modelInput')
    outputs = biggerModel(inputs, outc=18 + 1, training=True)
    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    model.save_weights('models/test/test')
    # model.load_weights('models/test/test')

    print(model.get_layer('quater').get_weights()[0][0][0][0:4])

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs = tf.keras.Input(shape=(224, 224, 3), name='modelInput')
    outputs = biggerModel(inputs, outc=18 + 1, training=False)
    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    # model.save_weights('models/test/test')
    model.load_weights('models/test/test')

    print(model.get_layer('quater').get_weights()[0][0][0][0:4])

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs = tf.keras.Input(shape=(224, 224, 3), name='modelInput')
    outputs = smallModel(inputs, outc=18 + 1, training=True)
    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    # model.save_weights('models/test/test')
    model.load_weights('models/test/test')

    print(model.get_layer('quater').get_weights()[0][0][0][0:4])

if __name__ == '__main__':

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs = tf.keras.Input(shape=(224, 224, 3), name='modelInput')
    outputs = smallModel(inputs, outc=18 + 1, training=False)
    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    # model.save_weights('models/test/test')
    model.load_weights('models/test/test')

    print(model.get_layer('quater').get_weights()[0][0][0][0:4])

def smallModel(inputs, outc, training=True):

    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(inputs)
    quater_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    quater_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='quater', activation=None)(quater_res)

    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(quater_res)
    half_res = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)(half_res)
    half_res_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='half', activation=None)(half_res)

    if training:
        return quater_res_out, half_res_out
    else:
        return quater_res_out

def create_checkpoint(model):
  return tf.train.Checkpoint(**{layer.name: layer for layer in model.layers})

model = ...
ckpt = create_checkpoint(model)
ckpt_path = ckpt.save("/path/to/ckpt")

# loading
model = ...
ckpt = create_checkpoint(model)
ckpt.restore(ckpt_path)