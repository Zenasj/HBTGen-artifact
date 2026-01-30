import random
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


m1 = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = m1(noise, training=False)
print([v.name for v in m1.trainable_variables])

m2 = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = m2(noise, training=False)
print([v.name for v in m2.trainable_variables])

with tf.Graph().as_default():
    m1 = make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = m1(noise, training=False)
    print([v.name for v in m1.trainable_variables])

with tf.Graph().as_default():
    m2 = make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = m2(noise, training=False)
    print([v.name for v in m2.trainable_variables])

a = tf.Variable(1, name='test')
b = tf.Variable(1, name='test')
print(a)
print(b)

import tensorflow as tf


dense1 = tf.keras.layers.Dense(10)
var1 = tf.Variable([10])

with tf.name_scope('scoped'):
    dense2 = tf.keras.layers.Dense(15)
    var2 = tf.Variable([15])

dense1(tf.ones([10, 15]))
dense2(tf.ones([10, 20]))

print(dense1.variables[0].name)
print(var1.name)
print(dense2.variables[0].name)
print(var2.name)

import tensorflow as tf


dense = tf.keras.layers.Dense(10)
var1 = tf.Variable([10])

with tf.name_scope('scoped'):
    dense1 = tf.keras.layers.Dense(15)
    var2 = tf.Variable([15])
    dense2 = tf.keras.layers.Dense(15)
    dense3 = tf.keras.layers.Dense(15)
    dense3.build((15,))

dense(tf.ones([10, 15]))
dense1(tf.ones([10, 20]))

with tf.name_scope('scoped'):
    dense2(tf.ones([10, 20]))

print(dense.variables[0].name)
print(var1.name)
print(dense1.variables[0].name) # Created in tf.name_scope('scoped') & First called in root name_scope
print(var2.name)
print(dense2.variables[0].name) # Created in tf.name_scope('scoped') & First called in tf.name_scope('scoped')
print(dense3.variables[0].name) # Called .build on it within tf.name_scope('scoped')

dense = tf.keras.layers.Dense(15)
dense(tf.ones([10, 20]))

with tf.name_scope('first'):
    denseN = tf.keras.layers.Dense(15)

with tf.name_scope('scoped'):
    denseN(tf.ones([10, 20]))
    
print(dense.variables[0].name)
print(denseN.variables[0].name)

train_model = MyModel()

train_one_epoch(train_model)
checkpoint_path = save_checkpoint(train_model)

eval_model = MyModel()
restore(checkpoint_path, eval_model) # This will fail due to the second call to MyModel() which creates different "unique" layer names.
eval(eval_model)

train_one_epoch(train_model)