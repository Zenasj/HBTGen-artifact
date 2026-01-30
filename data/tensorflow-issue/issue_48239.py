import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

py
import tensorflow as tf
import numpy as np

from os.path import join as opj

from collections import namedtuple


def pseudo_derivative(v_scaled, dampening_factor):
    '''
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    '''
    return dampening_factor*tf.maximum(1 - tf.abs(v_scaled), 0)
    #return dampening_factor*tf.exp(-2.0*tf.abs(v_scaled))


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    '''
    The tensorflow function which is defined as a Heaviside function (to compute the spikes),
    but with a gradient defined with the pseudo derivative.
    :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
    :param dampening_factor: parameter to stabilize learning
    :param derivative_width: parameter which will rescale the width of the pseudo-derivative
    :return: the spike tensor
    '''
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    grad = lambda dy: [dy*pseudo_derivative(v_scaled, dampening_factor), tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad

@tf.function
def voltage_reg_loss(v_scaled):

    reg_thresh_pos = 0.4
    reg_thresh_neg = -2.0

    per_neuron_time_step = tf.nn.relu(v_scaled - 0.4)**2 + tf.nn.relu(-2.0 - v_scaled)**2

    loss_volt = tf.math.reduce_mean(per_neuron_time_step)**2

    return loss_volt


class LIFCell(tf.keras.layers.Layer):

    def __init__(self,
                 n_rec,
                 tau=20.,
                 thr=1.,
                 dt=1,
                 n_refractory=5,
                 dampening_factor=.3):

        super().__init__(self)

        self.n_rec = n_rec

        self._dt = float(dt)
        self._decay = tf.exp(-dt / tau)
        self._n_refractory = n_refractory


        self.input_weights = None
        self.bias_currents = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        self.threshold = thr
        self._dampening_factor = dampening_factor

    def build(self, input_shape):

        initializer = tf.keras.initializers.GlorotNormal()

        self.input_weights = self.add_weight(shape=(input_shape[-1], self.n_rec),
                                             initializer=initializer,
                                             name='input_weights')

        self.disconnect_mask = tf.cast(np.diag(np.ones(self.n_rec, dtype=np.bool)), tf.bool)

        self.recurrent_weights = self.add_weight(shape=(self.n_rec, self.n_rec),
                                                 initializer=initializer,
                                                 name='recurrent_weights')

    @property
    def state_size(self):
        return self.n_rec, self.n_rec, self.n_rec

    @property
    def output_size(self):
        return self.n_rec, self.n_rec, self.n_rec

    def zero_state(self, batch_size, dtype=tf.float32):
        v0 = tf.zeros((batch_size, self.n_rec), dtype)
        r0 = tf.zeros((batch_size, self.n_rec), tf.int32)
        z_buf0 = tf.zeros((batch_size, self.n_rec), tf.float32)
        return v0, r0, z_buf0

    def get_initial_state(self, batch_size, inputs, dtype=tf.float32):
        v0 = tf.zeros((batch_size, self.n_rec), dtype)
        r0 = tf.zeros((batch_size, self.n_rec), tf.int32)
        z_buf0 = tf.zeros((batch_size, self.n_rec), tf.float32)
        return v0, r0, z_buf0

    @tf.function
    def call(self, inputs, initial_state):
        old_v = initial_state[0]
        old_r = initial_state[1]
        old_z = initial_state[2]

        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, no_autapse_w_rec)

        i_reset = -self.threshold * old_z
        input_current = (1.0 - self._decay)*(i_in + i_rec) + i_reset

        new_v = self._decay * old_v + input_current

        is_refractory = tf.greater(old_r, 0)
        v_scaled = (new_v - self.threshold) / self.threshold
        new_z = spike_function(v_scaled, self._dampening_factor)
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self._n_refractory, tf.int32), 0, self._n_refractory)

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state


def create_pretrain_model(cell, seq_len=1000, n_input=40):

    inputs = tf.keras.layers.Input(shape=(seq_len, n_input))
    batch_size = tf.shape(inputs)[0]

    rnn = tf.keras.layers.RNN(cell, return_sequences=True)

    initial_state = cell.zero_state(batch_size)
    v, z = rnn(inputs, initial_state=initial_state)

    return tf.keras.Model(inputs=inputs, outputs=[v, z])


if __name__ == '__main__':

    with tf.device('/GPU:0'):

        # initialize the cell
        cell = LIFCell(512)

        # create the model with some inputs
        model = create_pretrain_model(cell, seq_len=1000, n_input=40)

        # save initial weights
        model.save_weights(opj('ckpt', 'init_weights'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # optimize the network simply so that "voltages" tend to lie within a certain range
        @tf.function
        def pre_train_step(samples):

            with tf.GradientTape() as tape:

                v, z = model(samples)

                scaled_voltage = (v - 1.0)

                loss = voltage_reg_loss(scaled_voltage)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return v, z

        # only 10 iterations
        for i in range(10):

            # input is irrelevant for reproducing the bug
            samples = tf.random.uniform((100, 1000, 40))

            # training step
            v, z = pre_train_step(samples)

        # save pre-trained weights
        model.save_weights(opj('ckpt', 'pretrain_weights'))