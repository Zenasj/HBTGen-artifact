import random
from tensorflow.keras import layers

# this code does not make much sense, but is shorter than providing a full optimization loop for a WGAN_GP and produces the same error
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as kl
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def gradient_penalty(model, input_data):
    # get gradient
    input_data = tf.convert_to_tensor(input_data)
    with tf.GradientTape() as t:
        t.watch(input_data)
        pred = model(input_data)
    grad = t.gradient(pred, [input_data])[0]
    # define gradient penalty
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp

if __name__ == "__main__":
    # model with recurrent layer
    model = k.Sequential([kl.InputLayer(input_shape=(50, 20)), kl.GRU(100), kl.Dense(1)])
    # Optimizer
    opt = tf.optimizers.Adam()
    # Dummy data
    data = np.random.normal(0, 1, (8, 50, 20)).astype(np.float32)
    # Optimize
    with tf.GradientTape() as tape:
        gp = gradient_penalty(model=model, input_data=data)
    grad = tape.gradient(gp, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))