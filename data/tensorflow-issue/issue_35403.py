from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import layers

def seq(k):
    result = tf.keras.Sequential(name='seq'+str(k))
    result.add(tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization(name='bn_'+str(k)))
    result.add(tf.keras.layers.GaussianNoise(stddev=1, name='noise_'+str(k)))
    result.add(tf.keras.layers.ReLU())

    return result


def testModel():
    input_layer = layers.Input(shape=(256, 256, 3))
    l0 = seq(0)(input_layer)
    l1 = seq(1)(l0)
    l2 = seq(2)(l1)
    model = tf.keras.Model(input_layer, l2)
    return model


if __name__ == '__main__':
    model = testModel()
    model.summary()
    seq0 = model.get_layer('seq_0')
    bn0 = seq0.get_layer('bn_0').output
    noise0 = seq0.get_layer('noise_0').output

    sub0 = tf.keras.layers.Subtract()([noise0, bn0])

    new_model = tf.keras.Model(model.input, [sub0, model.output])

    test_tensor = tf.ones(shape=(1, 256, 256, 3))

    (out, z) = new_model(test_tensor)

import tensorflow as tf
from tensorflow.keras import layers

def seq_model(k):
    model = tf.keras.Sequential(name='seq_'+str(k))
    model.add(layers.Conv2D(64, (3, 3), strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization(name='bn_'+str(k)))
    model.add(layers.GaussianNoise(stddev=1, name='noise_'+str(k)))
    model.add(layers.ReLU())

    return model

def mymodel():
    input = tf.keras.Input(shape=(224, 224, 3))

    s0 = seq_model(k=0)(input)
    s1 = seq_model(k=1)(s0)
    s2 = seq_model(k=2)(s1)

    return tf.keras.Model(input, s2)

model = mymodel()
noise_0 = model.get_layer('seq_0').get_layer('noise_0').output
bn_0 = model.get_layer('seq_0').get_layer('bn_0').output

sub_0 = layers.Subtract()([noise_0, bn_0])
new_model = tf.keras.Model(model.input, [model.output, sub_0])

test_tensor = tf.zeros(shape=(1, 224, 224, 3))
out_of_new_model = new_model(test_tensor, training=True)