# tf.random.uniform((B, T, D_in), dtype=tf.float32) ← typical input shape for LSTM inputs (batch, time, features)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, RNN, InputSpec
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops, math_ops, nn
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import control_flow_util
from tensorflow.python.eager import context

# Adapted from TensorFlow 1.14 LSTMCell with step counter in states & optional recurrent batch normalization.
# This is a fused model (cell+wrapper) representing the core idea in the issue with step counter in states.

# Helper to return caching device compatible with graph/eager mode
def _caching_device(rnn_cell):
    if context.executing_eagerly():
        return None
    if not getattr(rnn_cell, '_enable_caching_device', False):
        return None
    if control_flow_util.IsInWhileLoop(ops.get_default_graph()):
        return None
    if rnn_cell._dtype_policy.should_cast_variables:
        return None
    return lambda op: op.device


class MyModel(tf.keras.Model):
    def __init__(self, units=32, rbn_configs=None, max_inference_steps=20, **kwargs):
        super().__init__(**kwargs)
        # Instantiate the custom LSTM layer with counter in states, optionally recurrent batch norm
        self.lstm = LSTM(
            units=units,
            rbn_configs=rbn_configs or {},
            max_inference_steps=max_inference_steps,
            return_sequences=False,  # return single output (last)
            return_state=False
        )

    def call(self, inputs, training=None):
        return self.lstm(inputs, training=training)


class LSTMCell(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 rbn_configs={},
                 max_inference_steps=20,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        if self.recurrent_dropout != 0 and implementation != 1:
            # Enforce implementation 1 (dynamic loop) if recurrent dropout or rbn active
            self.implementation = 1
        elif rbn_configs and implementation != 1:
            self.implementation = 1
        else:
            self.implementation = implementation

        # State size includes h, c, and the step counter as a scalar int64
        self.state_size = [self.units, self.units, 1]
        self.output_size = self.units

        self.recurrent_batchnorm = bool(rbn_configs)
        self.rbn_configs = rbn_configs or {}
        self.max_inference_steps = max_inference_steps
        self.max_inference_step = K.constant(max_inference_steps - 1, dtype='int64')
        self._decay = K.constant(1.0 - self.rbn_configs.get('bn_momentum', 0.99),
                                 dtype='float32')

        if self.recurrent_batchnorm:
            self.__init_recurrent_batchnorm(self.rbn_configs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        default_caching_device = _caching_device(self)

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None

        if self.recurrent_batchnorm:
            self.build_recurrent_bn(input_shape)

        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        step = states[2]   # scalar int64 step counter

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs_f = inputs_c = inputs_o = inputs

        k_i, k_f, k_c, k_o = array_ops.split(self.kernel, 4, axis=1)
        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)
        x_c = K.dot(inputs_c, k_c)
        x_o = K.dot(inputs_o, k_o)

        if self.recurrent_batchnorm:
            x_i = self.recurrent_bn(x_i, 'kernel_i', step, training)
            x_f = self.recurrent_bn(x_f, 'kernel_f', step, training)
            x_c = self.recurrent_bn(x_c, 'kernel_c', step, training)
            x_o = self.recurrent_bn(x_o, 'kernel_o', step, training)

        if self.use_bias:
            b_i, b_f, b_c, b_o = array_ops.split(self.bias, 4, axis=0)
            x_i = K.bias_add(x_i, b_i)
            x_f = K.bias_add(x_f, b_f)
            x_c = K.bias_add(x_c, b_c)
            x_o = K.bias_add(x_o, b_o)

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1_f = h_tm1_c = h_tm1_o = h_tm1

        x = (x_i, x_f, x_c, x_o)
        h_tm1_tuple = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
        c, o = self._compute_carry_and_output(x, h_tm1_tuple, c_tm1, step, training)
        h = o * self.activation(c)
        # Increment step counter (int64 scalar)
        next_step = step + 1
        return h, [h, c, next_step]

    def _compute_carry_and_output(self, x, h_tm1, c_tm1, step, training):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1

        z0 = K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units])
        z1 = K.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2])
        z2 = K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3])
        z3 = K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:])

        if self.recurrent_batchnorm:
            z0 = self.recurrent_bn(z0, 'recurrent_i', step, training)
            z1 = self.recurrent_bn(z1, 'recurrent_f', step, training)
            z2 = self.recurrent_bn(z2, 'recurrent_c', step, training)
            z3 = self.recurrent_bn(z3, 'recurrent_o', step, training)

        i = self.recurrent_activation(x_i + z0)
        f = self.recurrent_activation(x_f + z1)
        c = f * c_tm1 + i * self.activation(x_c + z2)
        o = self.recurrent_activation(x_o + z3)

        if self.recurrent_batchnorm:
            c = self.recurrent_bn(c, 'cell', step, training)

        return c, o

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None or dtype is None:
            raise ValueError("'batch_size and dtype cannot be None.")

        # Create zeros for h and c states, and scalar zero int64 for step counter
        def create_zeros(size):
            dims = tensor_shape.as_shape(size).as_list()
            if len(dims) == 1 and dims[0] == 1:
                return tf.constant(0, dtype='int64')
            else:
                return tf.zeros([batch_size] + dims, dtype=dtype)

        return nest.map_structure(create_zeros, self.state_size)

    def get_dropout_mask_for_cell(self, inputs, training, count):
        if self.dropout <= 0:
            return [None] * count
        return _generate_dropout_mask(tf.ones_like(inputs), self.dropout, training, count)

    def get_recurrent_dropout_mask_for_cell(self, h_tm1, training, count):
        if self.recurrent_dropout <= 0:
            return [None] * count
        return _generate_dropout_mask(tf.ones_like(h_tm1), self.recurrent_dropout, training, count)

    def recurrent_bn(self, inputs_t, op_name, step, training=None):
        training = self._get_training_value(training)
        beta, gamma, moving_mean, moving_variance = self._get_bn_vars(op_name)

        @tf.function
        def compute_bn(inputs_t, step, beta, gamma, moving_mean, moving_variance, training):
            # Use moving averages or update them if training and step < max_inf_step
            def true_fn():
                mean_t, variance_t = self._get_stats_and_maybe_update(
                    inputs_t, step, moving_mean, moving_variance, training)
                return nn.batch_normalization(inputs_t, mean_t, variance_t,
                                             beta, gamma, self.rbn_configs.get('bn_epsilon', 1e-3))

            def false_fn():
                return nn.batch_normalization(inputs_t,
                                             moving_mean[-1], moving_variance[-1],
                                             beta, gamma, self.rbn_configs.get('bn_epsilon', 1e-3))

            cond = tf.math.logical_and(training, tf.math.less(step, self.max_inference_step))
            return tf_utils.smart_cond(cond, true_fn, false_fn)

        outputs = compute_bn(inputs_t, step, beta, gamma, moving_mean, moving_variance, training)
        outputs.set_shape(inputs_t.shape)
        return outputs

    def _get_bn_vars(self, op_name):
        return [getattr(self, op_name + '_' + var_name)
                for var_name in ('gamma', 'beta', 'moving_mean', 'moving_variance')]

    def _get_stats_and_maybe_update(self, inputs_t, step, moving_mean, moving_variance, training):
        moving_mean_t = moving_mean[step:step + 1]
        moving_variance_t = moving_variance[step:step + 1]

        return tf_utils.smart_cond(training,
                                   lambda: self._update_moving_avgs_and_get_batch_stats(
                                       inputs_t, moving_mean_t, moving_variance_t, training),
                                   lambda: (moving_mean_t, moving_variance_t))

    def _update_moving_avgs_and_get_batch_stats(self, inputs_t, moving_mean_t, moving_variance_t, training):
        mean_t, variance_t = nn.moments(inputs_t, axes=[0], keepdims=False)
        self._assign_moving_average(moving_mean_t, mean_t)
        self._assign_moving_average(moving_variance_t, variance_t)
        return mean_t, variance_t

    def _assign_moving_average(self, variable, value):
        with K.name_scope('AssignMovingAvg'):
            with ops.colocate_with(variable):
                update_delta = (variable - tf.cast(value, variable.dtype)) * self._decay
                variable.assign(variable - update_delta)

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        return tf.cast(training, tf.bool)

    def build_recurrent_bn(self, input_shape):
        default_caching_device = _caching_device(self)

        def _build_steps_variable(var_fullname, fullname):
            return self.add_weight(
                shape=(self.max_inference_steps, self.units),
                name=fullname,
                initializer=getattr(self, var_fullname + '_initializer'),
                trainable=False,
                caching_device=default_caching_device)

        def _build_variable(var_fullname, fullname):
            return self.add_weight(
                shape=(1, self.units),
                name=fullname,
                initializer=getattr(self, var_fullname + '_initializer'),
                regularizer=getattr(self, var_fullname + '_regularizer'),
                constraint=getattr(self, var_fullname + '_constraint'),
                trainable=True,
                caching_device=default_caching_device)

        for weight_name in ('kernel', 'recurrent', 'cell'):
            gate_names = ('_i', '_f', '_c', '_o') if weight_name != 'cell' else ('',)
            for gate_name in gate_names:
                op_name = weight_name + gate_name
                for var_name in ('gamma', 'beta', 'moving_mean', 'moving_variance'):
                    fullname = op_name + '_' + var_name
                    var_fullname = weight_name + '_' + var_name

                    if 'moving' in var_name:
                        setattr(self, fullname, _build_steps_variable(var_fullname, fullname))
                    else:
                        setattr(self, fullname, _build_variable(var_fullname, fullname))

    def __init_recurrent_batchnorm(self, configs):
        default_configs = dict(
            steps_in=None,
            max_inference_steps_frac=1,
            bn_epsilon=1e-3,
            bn_momentum=0.99,
            kernel_gamma_initializer=0.1,
            kernel_gamma_regularizer=None,
            kernel_gamma_constraint=None,
            kernel_beta_initializer='zeros',
            kernel_beta_regularizer=None,
            kernel_beta_constraint=None,
            kernel_moving_mean_initializer='zeros',
            kernel_moving_variance_initializer=0.1,
            recurrent_gamma_initializer=0.1,
            recurrent_gamma_regularizer=None,
            recurrent_gamma_constraint=None,
            recurrent_beta_initializer='zeros',
            recurrent_beta_regularizer=None,
            recurrent_beta_constraint=None,
            recurrent_moving_mean_initializer='zeros',
            recurrent_moving_variance_initializer=0.1,
            cell_gamma_initializer=0.1,
            cell_gamma_regularizer=None,
            cell_gamma_constraint=None,
            cell_beta_initializer='zeros',
            cell_beta_regularizer=None,
            cell_beta_constraint=None,
            cell_moving_mean_initializer='zeros',
            cell_moving_variance_initializer=0.1,
        )
        for key, val in default_configs.items():
            setattr(self, key, configs.get(key, val))
        # max_inference_steps calculated externally in constructor

class LSTM(RNN):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 rbn_configs={},
                 max_inference_steps=20,
                 **kwargs):
        if implementation == 0:
            implementation = 1  # deprecated

        self.cell = LSTMCell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            rbn_configs=rbn_configs,
            max_inference_steps=max_inference_steps
        )
        super().__init__(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = [InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        return super().call(inputs, mask=mask, training=training, initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(dropped_inputs, ones, training=training) for _ in range(count)]
    return K.in_train_phase(dropped_inputs, ones, training=training)


def my_model_function():
    # Return an instance of MyModel with default params
    return MyModel()

def GetInput():
    # Return a random tensor input of shape (batch=2, time=5, features=10)
    # matching the default input expected by MyModel (with units=32)
    batch_size = 2
    time_steps = 5
    features = 10
    return tf.random.uniform(shape=(batch_size, time_steps, features), dtype=tf.float32)

