import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

for step in range(settings['iterations']):
    x_train, y_train = batch_fn()
    with tf.GradientTape() as tape:
        logits, probs = model(x_train)
        loss_value = loss(y_train, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

import tensorflow as tf

class SegmentedMean(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SegmentedMean, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features, segments = inputs
        return tf.math.segment_mean(features, segments)

inputs = tf.keras.Input(shape=(feats_len,), name='features')
segments = tf.keras.Input(shape=(), name='segments', dtype=tf.int32)
x = tf.keras.layers.Dense(settings['k'], activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(settings['k'])(x)

x = SegmentedMean()((x, segments))
x = tf.keras.layers.Dense(settings['k'], activation=tf.nn.relu)(x)
logits = tf.keras.layers.Dense(2, name='output_logits')(x)
probs = tf.keras.layers.Softmax()(logits)
model = tf.keras.Model(inputs=(inputs, segments), outputs=(logits, probs), name='mil_model')
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss={'output_logits': loss})
model.fit_generator(generator, epochs=5, workers=4)