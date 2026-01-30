import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

H, W, C = 128, 128, 9
imgs = tf.zeros([10, H, W, C])
ds = tf.data.Dataset.from_tensor_slices(imgs).batch(2)
print(ds)


def construstor(inputs):
    shortcuts = []
    weights_initializer = tf.compat.v1.initializers.truncated_normal(
        mean=0.0,
        stddev=tf.math.sqrt(
            2.0 / ((3 ** 2) * 32))
    )
    activation_fn = tf.keras.layers.LeakyReLU()

    def _operation_convolutional(_filters):
        return tf.keras.layers.Conv2D(
            filters=_filters,
            strides=[1, 1],
            activation=activation_fn,
            kernel_initializer=weights_initializer,
            kernel_size=[3, 3],
            padding='SAME',
            kernel_regularizer=tf.keras.regularizers.l2(1.0),
            bias_regularizer=tf.keras.regularizers.l2(1.0))

    outputs = _operation_convolutional(32)(inputs)
    return _operation_convolutional(3)(outputs)

input_tensor = tf.keras.Input(shape=[None, None, 3])
model = tf.keras.Model(inputs=input_tensor,
                       outputs=construstor(input_tensor))
def run(img):
    tf.summary.image('img', img)

writer = tf.summary.create_file_writer(r"D:\tmp")
with writer.as_default():
    for i, img in enumerate(ds):
        tf.summary.experimental.set_step(i)
        print('iteration')
        outputs = model(img)
        run(outputs)