# tf.random.uniform((B, 1000, 40), dtype=tf.float32)

import tensorflow as tf
import numpy as np

def pseudo_derivative(v_scaled, dampening_factor):
    '''
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    '''
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)
    # Alternative (commented out):
    # return dampening_factor * tf.exp(-2.0 * tf.abs(v_scaled))


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    '''
    The tensorflow function defined as a Heaviside step (to compute spikes),
    but with a gradient defined with the pseudo derivative.
    :param v_scaled: scaled voltage: -1 at rest, 0 at threshold
    :param dampening_factor: parameter to stabilize learning
    :return: spike tensor
    '''
    z_ = tf.cast(tf.greater(v_scaled, 0.), dtype=tf.float32)

    def grad(dy):
        # Gradient defined by pseudo derivative for v_scaled, zero gradient for dampening_factor
        return dy * pseudo_derivative(v_scaled, dampening_factor), tf.zeros_like(dampening_factor)

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.function
def voltage_reg_loss(v_scaled):
    '''
    Voltage regularization loss to encourage voltage to stay in range.
    Uses squared ReLU penalties outside thresholds.
    :param v_scaled: scaled voltage tensor
    :return: scalar loss value
    '''
    reg_thresh_pos = 0.4
    reg_thresh_neg = -2.0

    per_neuron_time_step = tf.nn.relu(v_scaled - reg_thresh_pos)**2 + tf.nn.relu(reg_thresh_neg - v_scaled)**2
    loss_volt = tf.math.reduce_mean(per_neuron_time_step)**2
    return loss_volt


class LIFCell(tf.keras.layers.Layer):
    '''
    Leaky Integrate-and-Fire (LIF) spiking neuron cell as a Keras RNN cell.
    '''
    def __init__(self,
                 n_rec,
                 tau=20.,
                 thr=1.,
                 dt=1,
                 n_refractory=5,
                 dampening_factor=0.3):
        super().__init__()

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
        # Disconnect self-connections (autapses)
        self.disconnect_mask = tf.cast(np.diag(np.ones(self.n_rec, dtype=np.bool_)), tf.bool)
        self.recurrent_weights = self.add_weight(shape=(self.n_rec, self.n_rec),
                                                 initializer=initializer,
                                                 name='recurrent_weights')

    @property
    def state_size(self):
        # (v, r, z_buf)
        return (self.n_rec, self.n_rec, self.n_rec)

    @property
    def output_size(self):
        # Output voltage and spikes
        return (self.n_rec, self.n_rec)

    def zero_state(self, batch_size, dtype=tf.float32):
        v0 = tf.zeros((batch_size, self.n_rec), dtype)
        r0 = tf.zeros((batch_size, self.n_rec), tf.int32)  # refractory counters
        z_buf0 = tf.zeros((batch_size, self.n_rec), tf.float32)
        return (v0, r0, z_buf0)

    def get_initial_state(self, batch_size, inputs=None, dtype=tf.float32):
        return self.zero_state(batch_size, dtype=dtype)

    @tf.function
    def call(self, inputs, initial_state):
        '''
        Run a single time step of the LIF cell.
        :param inputs: shape (batch_size, input_dim)
        :param initial_state: tuple of (v, refractory counter r, previous spike z_buf)
        :return: output (new_v, new_z), new_state
        '''
        old_v, old_r, old_z = initial_state

        # Zero out recurrent weights for self connections
        no_autapse_w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)

        # Input current from inputs and recurrent spikes
        i_in = tf.matmul(inputs, self.input_weights)
        i_rec = tf.matmul(old_z, no_autapse_w_rec)

        # Reset input for spiking neurons that spiked previously
        i_reset = -self.threshold * old_z

        # Calculate total input current with decay
        input_current = (1.0 - self._decay) * (i_in + i_rec) + i_reset

        # Update membrane potential voltage
        new_v = self._decay * old_v + input_current

        # Refractory neurons do not spike
        is_refractory = tf.greater(old_r, 0)

        # Scale voltage for spike thresholding (-1 at rest, 0 at threshold)
        v_scaled = (new_v - self.threshold) / self.threshold

        # Compute spikes with custom gradient function
        new_z = spike_function(v_scaled, self._dampening_factor)
        # Zero spike output for refractory neurons
        new_z = tf.where(is_refractory, tf.zeros_like(new_z), new_z)

        # Update refractory counters: decrement by one and add refractory period for new spikes
        new_r = tf.clip_by_value(old_r - 1 + tf.cast(new_z * self._n_refractory, tf.int32), 0, self._n_refractory)

        new_state = (new_v, new_r, new_z)
        output = (new_v, new_z)

        return output, new_state


class MyModel(tf.keras.Model):
    '''
    Model wrapping the LIFCell in a Keras RNN for sequence inputs.
    Inputs: Tensor shape (batch_size, seq_len, n_input)
    Outputs: tuple of (voltages, spikes) both shape (batch_size, seq_len, n_rec)
    '''

    def __init__(self, n_rec=512, seq_len=1000, n_input=40):
        super().__init__()
        self.cell = LIFCell(n_rec=n_rec)
        self.seq_len = seq_len
        self.n_input = n_input
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        initial_state = self.cell.zero_state(batch_size)
        voltages, spikes = self.rnn(inputs, initial_state=initial_state)
        return voltages, spikes


def my_model_function():
    return MyModel()


def GetInput():
    # Generate a random input tensor shaped (batch_size, seq_len, n_input)
    # Using batch_size = 100, seq_len = 1000, n_input = 40 as per example
    batch_size = 100
    seq_len = 1000
    n_input = 40
    return tf.random.uniform((batch_size, seq_len, n_input), dtype=tf.float32)

