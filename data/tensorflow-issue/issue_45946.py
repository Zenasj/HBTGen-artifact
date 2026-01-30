import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import keras

IMAGE_SHAPE = [256,256,3]

def Discriminator():
    return keras.Sequential([
        keras.layers.Flatten(input_shape=IMAGE_SHAPE),
        keras.layers.Dense(1, activation="sigmoid")
    ])

def Generator():
    return keras.Sequential([
        keras.layers.Conv2D(filters=IMAGE_SHAPE[-1], kernel_size=3, strides=1, padding="same", use_bias=False,
                           input_shape=IMAGE_SHAPE)
    ])

generator_BtoA = Generator()
discriminator_A = Discriminator()

loss_obj = keras.losses.MeanSquaredError()

discriminator_A_optimizer = keras.optimizers.Adam(0.0002)

BATCH_SIZE = 32

@tf.function
def train_step():
    # training discriminator
    imagesA = tf.random.uniform([BATCH_SIZE]+IMAGE_SHAPE)
    imagesB = tf.random.uniform([BATCH_SIZE]+IMAGE_SHAPE)
    fakesA = generator_BtoA(imagesB, training=False)
    with tf.GradientTape(persistent=True) as tape:
        disc_fakesA = discriminator_A(fakesA, training=True)
        discA_loss = loss_obj(tf.zeros_like(disc_fakesA), disc_fakesA)
    gradients_discA = tape.gradient(discA_loss, discriminator_A.trainable_variables)
    discriminator_A_optimizer.apply_gradients(zip(gradients_discA, discriminator_A.trainable_variables))


from tensorflow.profiler.experimental import Trace as Trace_profiler, start as start_profiler, stop as stop_profiler

start_profiler("my_logdir/")
with Trace_profiler("train", step_num=1, _r=-1):
    train_step()
stop_profiler()