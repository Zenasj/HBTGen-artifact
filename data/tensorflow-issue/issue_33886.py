# tf.random.uniform((B, T, 28), dtype=tf.float32) ← MNIST data shaped as (batch, time_steps, input_dim) with input_dim=28 from code

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, RNN
from tensorflow.keras import activations, initializers, backend as K
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

class MyModel(tf.keras.Model):
    """
    This model implements a custom RNN using a specialized RNN cell (AutoconceptorCell),
    which augments the standard hidden state with an additional matrix state that is 
    initialized as an identity matrix per batch element.

    It is designed similar to the referenced GitHub issue:
    - The cell state carries both a vector hidden state and a matrix state.
    - The initial_state to the RNN layer must match batch dimensions.
    """
    def __init__(self, units=50, output_dim=10, ac_alpha=60, ac_lambda=0.105):
        super().__init__()
        self.units = units
        self.ac_alpha = ac_alpha
        self.ac_lambda = ac_lambda
        self.output_dim = output_dim

        self.rnn_layer = RNN(
            AutoconceptorCell(self.units, alpha=self.ac_alpha, lamb=self.ac_lambda),
            return_sequences=False,  # as in original code
            return_state=False
        )
        self.dense = Dense(self.output_dim, activation='softmax')

    def call(self, inputs):
        """
        Forward pass through the RNN layer with a custom initial state,
        followed by dense classification output.

        inputs shape: (batch, time_steps, input_dim)
        """
        batch_size = tf.shape(inputs)[0]

        # Initialize states for RNN cell per batch:
        # - zero vector of shape (batch, units)
        # - identity matrix of shape (batch, units, units)
        initial_state = [
            tf.zeros((batch_size, self.units), dtype=inputs.dtype),
            tf.eye(self.units, batch_shape=[batch_size], dtype=inputs.dtype)
        ]

        x = self.rnn_layer(inputs, initial_state=initial_state)
        output = self.dense(x)
        return output


class AutoconceptorCell(Layer):
    """
    Custom RNN cell extending Layer with two states:
    - h: hidden state vector of shape (batch, units)
    - C: conceptor matrix of shape (batch, units, units)

    Implements a specialized recurrent update that maintains and updates a matrix state
    in addition to a standard hidden state, per the GitHub issue description.
    """

    def __init__(self,
                 units,
                 alpha,
                 lamb,
                 activation='tanh',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='identity',
                 bias_initializer='glorot_uniform',
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.aperture_fact = alpha ** (-2)
        self.l = lamb

        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        # state_size is a list [units, (units, units)] but wrapped in NoDependency as original,
        # this tells RNN the shapes to expect for the states.
        self.state_size = NoDependency([self.units, TensorShape([self.units, self.units])])
        self.output_size = self.units

        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='kernel',
            initializer=self.kernel_initializer)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)

        self.bias = self.add_weight(
            shape=(self.units,),
            name='bias',
            initializer=self.bias_initializer)

        self.built = True

    def call(self, inputs, states):
        # states = [prev_h (batch, units), C (batch, units, units)]
        prev_h, C = states

        # Compute input influence: (batch, units)
        input_infl = K.bias_add(K.dot(inputs, self.kernel), self.bias)
        # Compute recurrent influence: (batch, units)
        recurrent_infl = K.dot(prev_h, self.recurrent_kernel)

        # Raw hidden activation before conceptor filtering
        h = self.activation(input_infl + recurrent_infl)

        # Update conceptor matrix C:
        # term: batch_dot(batch_dims, batch_dims) to produce batch matrix product for weighted outer product.
        diff = h - K.batch_dot(h, C, axes=(1, 1))  # (batch, units)
        diff_exp = tf.expand_dims(diff, 2)          # (batch, units, 1)
        h_exp = tf.expand_dims(h, 1)                 # (batch, 1, units)

        # Outer product: for each batch element (units x 1) * (1 x units) -> (units x units)
        outer = K.batch_dot(diff_exp, h_exp)        # (batch, units, units)

        # Update conceptor matrix with decay and aperture scaling
        C = C + self.l * (outer - self.aperture_fact * C)

        # Apply conceptor filter and layer norm, then activation again
        filtered_h = K.batch_dot(h, C, axes=(1, 1))  # (batch, units)
        filtered_h = self.layer_norm(filtered_h)
        h = self.activation(filtered_h)

        # Return output and new states
        return h, [h, C]


def my_model_function():
    # Create an instance with default parameters inferred from the issue config
    return MyModel()


def GetInput():
    # Return a random input tensor to the model
    # According to the original code MNIST pixels are used as inputs with shape (batch, 28, 28)
    # But in the model call the input dim expected given kernel shape is (batch, time_steps, input_dim)
    # The input_dim used in kernel was input_shape[-1], so 28,
    # and time_steps is 28 as well since MNIST images are 28x28
    # We produce batched input with batch_size=4 for example
    batch_size = 4
    time_steps = 28
    input_dim = 28

    # Random uniform float32 tensor mimicking normalized pixel intensities [0, 1]
    return tf.random.uniform((batch_size, time_steps, input_dim), dtype=tf.float32)

