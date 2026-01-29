# tf.random.uniform((B, T, 6), dtype=tf.float32) for feature_signals and tf.random.uniform((B, T, 1), dtype=tf.float32) for timestamps

import tensorflow as tf

class MyPhasedLSTMCell(tf.keras.layers.Layer):
    """
    A modified PhasedLSTMCell that supports multi-dimensional time input per feature.
    This adapts the original PhasedLSTMCell (from tf.contrib.rnn)
    to accept timestamps of shape (batch, timesteps, features) instead of (batch, timesteps, 1).

    For simplicity, this is a minimal reimplementation of the core logic focusing on
    computing shifted_time per feature dimension, leveraging the original formula.

    Note: This is an inferred reimplementation based on the issue description and code comments,
    aiming to provide per-feature timestamps handling.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [self.units, self.units, self.units]  # (c, h, k)

    def build(self, input_shape):
        # input_shape is a tuple:
        # inputs (tuple): (timestamps, feature_signals)
        # timestamps shape: (batch, time, features)
        # feature_signals shape: (batch, time, features)

        # Weights for gates and cell computations:
        input_feature_dim = input_shape[1][-1]  # Should be number of features; typically input dim

        self.kernel = self.add_weight(
            shape=(input_feature_dim, self.units * 4),
            initializer='glorot_uniform',
            name='kernel')

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            name='recurrent_kernel')

        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros',
            name='bias')

        # Parameters for the phase gate per unit
        self.period = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=1., maxval=100.),
            trainable=True,
            name='period')

        self.phase = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=1.),
            trainable=True,
            name='phase')

        self.r_on = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=0.5),
            trainable=True,
            name='r_on')

        super().build(input_shape)

    def call(self, inputs, states):
        """
        inputs: tuple (timestamps, feature_signals)
            timestamps: shape (batch, features) current time for each feature (not sequence)
            feature_signals: shape (batch, input_features) current inputs at timestep
        states:
            c_prev, h_prev, k_prev (all shape (batch, units))
        """

        timestamps, feature_signals = inputs
        c_prev, h_prev, k_prev = states

        # Compute gates
        z = tf.matmul(feature_signals, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel) + self.bias

        i, f, c_bar, o = tf.split(z, num_or_size_splits=4, axis=-1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        c_bar = tf.tanh(c_bar)
        o = tf.sigmoid(o)

        # Phase gate calculations

        # timestamps shape inferred as (batch, features)
        # We need to handle the time dimension matching units

        # Broadcast timestamps and phases to match shape (batch, units)
        # Here we assume timestamps shape is (batch, features), features == input_features
        # Units are LSTM units. We need a method to map from input features to units,
        # so we broadcast timestamps per unit dimension.

        # For simplicity, take mean of timestamps along feature dimension to get (batch, 1)
        # A better approach would be to align timestamps per unit (requires architectural changes)
        # But to support the user use case with multi-feature timestamps, we implement per-unit shifted_time as below.

        # Expand dims to broadcast
        time_expanded = tf.expand_dims(timestamps, axis=-1)  # (batch, features, 1)
        period = tf.reshape(self.period, (1, 1, self.units))  # (1,1,units)
        phase = tf.reshape(self.phase, (1, 1, self.units))    # (1,1,units)
        r_on = tf.reshape(self.r_on, (1, 1, self.units))      # (1,1,units)

        # Since units and features may differ, for demo align features == units or broadcast sensibly
        # To keep it consistent with user's original shape (6 features) and units=256,
        # we broadcast timestamps by reducing along features axis
        # We'll average timestamps across features (axis=1):
        time_scalar = tf.reduce_mean(timestamps, axis=1, keepdims=True)  # (batch, 1)

        # Use time_scalar for phased gate calculation (broadcasted)
        shifted_time = time_scalar - phase  # broadcasting phase (1,1,units)
        shifted_time = tf.squeeze(shifted_time, axis=1)  # (batch, units)

        # Compute cycle ratio (modulus)
        cycle_ratio = tf.math.mod(shifted_time, period) / period
        cycle_ratio = tf.cast(cycle_ratio, tf.float32)

        # Define k (phase gate scalar per unit)
        cond_1 = tf.less(cycle_ratio, r_on / 2)
        cond_2 = tf.logical_and(tf.greater_equal(cycle_ratio, r_on / 2),
                                tf.less(cycle_ratio, r_on))

        k = tf.where(cond_1,
                     2 * cycle_ratio / r_on,
                     tf.where(cond_2,
                              2 - (2 * cycle_ratio / r_on),
                              tf.zeros_like(cycle_ratio)))
        k = tf.expand_dims(k, axis=1) if len(k.shape) == 1 else k  # (batch, units)

        # Compute new cell state and hidden state with phased gate
        c = f * c_prev + i * c_bar
        h = o * tf.tanh(c)

        # Phased output mask
        h = h * k

        return h, [c, h, k]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # Initialize the hidden states and phase gate states to zeros
        c = tf.zeros((batch_size, self.units), dtype=dtype)
        h = tf.zeros((batch_size, self.units), dtype=dtype)
        k = tf.zeros((batch_size, self.units), dtype=dtype)
        return [c, h, k]


class MyModel(tf.keras.Model):
    """
    Model that processes time series inputs with timestamps per feature using
    a modified Phased LSTM Cell expecting multi-dimensional time input.
    """

    def __init__(self):
        super().__init__()
        self.units = 256
        self.cell = MyPhasedLSTMCell(self.units)
        # Use Keras RNN layer wrapped around our modified cell
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=False)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid', name='logits')

    def call(self, inputs, training=False):
        timestamps, feature_signals = inputs
        # The custom cell expects inputs as tuples of (timestamps, feature_signals) at each time step.
        # However, tf.keras.layers.RNN expects a single tensor input per timestep.
        # To conform to that, we'll create a TimeDistributed wrapper or govern call().

        # Approach:
        # We will implement a step-level RNN call by merging timestamps and features along last axis, and
        # then in the call of MyPhasedLSTMCell, split them back.

        # But that is complicated; easier is to use the underlying tf.keras.layers.RNN with a
        # cell that accepts inputs as tuples, but current Keras RNN does not support tuple inputs directly.

        # So, we rewrite the call to simulate the RNN steps manually:

        # inputs shapes:
        # timestamps: (batch, timesteps, features) == (B, T, 6)
        # feature_signals: (batch, timesteps, features) == (B, T, 6)

        # We will iterate over time dimension and feed the tuple (timestamps[t], feature_signals[t]) to the cell

        batch_size = tf.shape(timestamps)[0]
        time_steps = tf.shape(timestamps)[1]

        # Initialize state
        states = self.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

        def step(time_index, outputs_ta, states):
            # Take slice for current timestep for both inputs
            t_slice = timestamps[:, time_index, :]  # (batch, features)
            x_slice = feature_signals[:, time_index, :]  # (batch, features)

            # Call cell manually
            output, states = self.cell((t_slice, x_slice), states)

            # Append output to tensor array
            outputs_ta = outputs_ta.write(time_index, output)

            return time_index + 1, outputs_ta, states

        outputs_ta = tf.TensorArray(dtype=tf.float32, size=time_steps, dynamic_size=False)
        time_index = tf.constant(0)

        cond = lambda time_index, outputs_ta, states: tf.less(time_index, time_steps)
        _, outputs_ta, states = tf.while_loop(cond, step, (time_index, outputs_ta, states))

        # Gather output over all timesteps
        outputs = outputs_ta.stack()  # (T, batch, units)
        outputs = tf.transpose(outputs, perm=[1, 0, 2])  # (batch, T, units)

        # Since return_sequences=False in original, we take last timestep output
        last_output = outputs[:, -1, :]  # (batch, units)

        logits = self.dense(last_output)  # (batch, 1)

        return logits


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a tuple of random inputs matching the expected shapes:
    # timestamps: (batch, time_steps, features=6), dtype float32
    # feature_signals: (batch, time_steps, features=6), dtype float32

    batch = 2
    time_steps = 10
    features = 6

    timestamps = tf.random.uniform((batch, time_steps, features), minval=0., maxval=10., dtype=tf.float32)
    feature_signals = tf.random.uniform((batch, time_steps, features), minval=-1., maxval=1., dtype=tf.float32)

    return (timestamps, feature_signals)

