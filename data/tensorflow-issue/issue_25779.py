import random
from tensorflow.keras import layers

python
import numpy as np
import tensorflow as tf
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import Dense, Flatten, Reshape
from keras import Input, Model
from keras.layers import Dense, Flatten, Reshape


def get_limb_lengths(person, limbs):
    """
    :param person: [ J * 3 ]
    :param limbs: {tf.constant} [ (a, b), (b, c), ... ]
    :return:
    """
    person = tf.reshape(person, (-1, 3))
    distances = tf.map_fn(
        lambda limb: tf.sqrt(
            tf.reduce_sum(
                tf.square(person[limb[0]] - person[limb[1]]))),
        limbs,
        dtype=tf.float32
    )
    return distances


def mean_limb_length_per_sequence(y_true, y_pred, limbs):
    """
    :param y_true: (n_frames, x J * 3)
    :param y_pred: (n_frames, x J * 3)
    :param limbs: {tf.constant}
    :return:
    """
    diff = tf.map_fn(
        lambda x:
        tf.reduce_mean(
            tf.abs(get_limb_lengths(x[0], limbs) -
                   get_limb_lengths(x[1], limbs))),
        (y_true, y_pred),
        dtype=tf.float32
    )
    return diff


def mean_limb_length_on_batch(y_true, y_pred, limbs):
    """ This one is optimized for CMU-MoCap
    :param y_true: (batchsize x n_frames x J * 3)
    :param y_pred: (batchsize x n_frames x J * 3)
    :return:
    """
    loss = tf.map_fn(
        lambda x: tf.reduce_mean(
            mean_limb_length_per_sequence(x[0], x[1], limbs)),
        (y_true, y_pred),
        dtype=tf.float32
    )
    return tf.reduce_mean(loss)


def mean_limb_length(y_true, y_pred):
    """ This one is optimized for CMU-MoCap
    :param y_true: (batchsize x n_frames x J * 3)
    :param y_pred: (batchsize x n_frames x J * 3)
    :return:
    """
    limbs = tf.constant([
        (0, 1), (1, 2), (2, 3), (3, 4)
    ])
    return mean_limb_length_on_batch(y_true,
                                     y_pred,
                                     limbs)


bs = 512
f1 = 20
f2 = 2
J = 5
dim = 3
X = np.random.random((bs, f1, J * dim))
Y = np.random.random((bs, f2, J * dim))

def create_gen():
    while True:
        yield X, Y

gen = create_gen()

inputs = Input(shape=X.shape[1:])
x = Flatten()(inputs)
x = Dense(f2 * J * dim)(x)
x = Reshape((f2, J, dim))(x)

model = Model(inputs=inputs, outputs=x)
model.summary()

optimizer = 'adam'

model.compile(loss=mean_limb_length,
              metrics=[mean_limb_length],
              optimizer=optimizer)

# model.fit(X, Y)
model.fit_generator(generator=gen,
                    validation_data=gen,
                    steps_per_epoch=10,
                    validation_steps=3,
                    epochs=5)

fit_generator