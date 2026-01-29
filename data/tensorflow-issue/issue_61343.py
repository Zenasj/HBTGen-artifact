# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape for dense layer input (batch_size, 784)

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from keras import activations, constraints, initializers, regularizers, backend
from keras.engine.input_spec import InputSpec
from keras.dtensor import utils

# Because the original custom Dense layer was named `MyDense`, we adapt this inside `MyModel`.
# We also incorporate the specialized Residue Number System (RNS) based modular arithmetic
# logic in the call to the layer that tries to compute outputs differently using mod divmod.

class MyDense(Layer):
    """Custom Dense layer with Residue Number System modular arithmetic computations."""

    @utils.allow_initializer_layout
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        n=6,  # added n parameter for modular base exponent (default 6 as in provided code)
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.n = n  # store n for modular arithmetic base exponent

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        self.w = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                "bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.b = None
        self.built = True

    def call(self, inputs):
        # Cast inputs to compute dtype if needed
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # Handle RaggedTensors (not fully expanded here, assuming flat input)
        # Omitted detailed ragged logic for brevity, assume dense tensor input.

        rank = inputs.shape.rank
        n = self.n

        if rank == 2 or rank is None:
            # Use the custom RNS modular arithmetic path inspired from original code:
            # The original issue code tried to do this custom divmod with powers of two and residues.

            # For clarity, rename 'self.w' to use convention w and 'self.b' to b:
            w = self.w
            b = self.b if self.b is not None else 0.0

            # Moduli for RNS:
            M1 = 2 ** n
            M2 = 2 ** n - 1
            M3 = 2 ** n + 1

            # Compute inputs mod each base
            _, x = tf.math.floordiv_mod(inputs, M1)  # x = inputs % M1
            _, x1 = tf.math.floordiv_mod(inputs, M2)  # x1 = inputs % M2
            _, x2 = tf.math.floordiv_mod(inputs, M3)  # x2 = inputs % M3

            # Compute weights mod each base
            _, w0 = tf.math.floordiv_mod(w, M1)  # w mod M1
            _, w1 = tf.math.floordiv_mod(w, M2)  # w mod M2
            _, w2 = tf.math.floordiv_mod(w, M3)  # w mod M3

            # Compute (x @ w + b) mod each modulus:
            # Here we respect matrix multiplication between inputs mod base with weights mod base:
            # Note: b broadcast over batch dimension and units dimension.

            z_mod_1 = tf.math.floormod(tf.matmul(x, w0) + b, M1)
            z_mod_2 = tf.math.floormod(tf.matmul(x1, w1) + b, M2)
            z_mod_3 = tf.math.floormod(tf.matmul(x2, w2) + b, M3)

            # Compute constants for reconstruction:
            Dm = M1 * M2 * M3
            m1 = Dm // M1
            m2 = Dm // M2
            m3 = Dm // M3

            # Modular multiplicative inverses for m1 mod M1, m2 mod M2, m3 mod M3:
            x1_inv = 1
            for i in range(1, M1):
                if (i * m1) % M1 == 1:
                    x1_inv = i
                    break
            x2_inv = 1
            for i in range(1, M2):
                if (i * m2) % M2 == 1:
                    x2_inv = i
                    break
            x3_inv = 1
            for i in range(1, M3):
                if (i * m3) % M3 == 1:
                    x3_inv = i
                    break

            # Use CRT (Chinese Remainder Theorem) to reconstruct decimal from residues:
            term1 = z_mod_1 * m1 * x1_inv
            term2 = z_mod_2 * m2 * x2_inv
            term3 = z_mod_3 * m3 * x3_inv
            num = tf.math.floormod(term1 + term2 + term3, Dm)

            outputs = tf.cast(num, self.dtype)

        else:
            # For input rank > 2 fallback to standard dense matmul:
            outputs = tf.tensordot(inputs, self.w, [[rank - 1], [0]])
            if self.b is not None:
                outputs = tf.nn.bias_add(outputs, self.b)

        # Apply bias if use_bias is True and wasn't applied above:
        if rank == 2 and self.b is not None:
            # Bias was included in RNS path,
            # so no extra addition needed here for rank==2.
            pass
        elif self.b is not None:
            outputs = tf.nn.bias_add(outputs, self.b)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the input shape of a Dense layer "
                "should be defined. Found None. "
                f"Received: input_shape={input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "n": self.n,
            }
        )
        return config


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define two dense layers:  
        # - one custom MyDense layer with RNS logic
        # - one standard keras Dense layer with activation relu for demonstration

        self.my_dense = MyDense(512, activation='relu', input_shape=(784,))
        self.dense_ref = keras.layers.Dense(512, activation='relu')

        # A dropout and output layer
        self.dropout = keras.layers.Dropout(0.2)
        self.out = keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # Pass through layers sequentially
        x1 = self.my_dense(inputs)       # custom dense
        x2 = self.dense_ref(inputs)      # standard dense for comparison

        # Compare outputs with a tolerance and return boolean tensor of elementwise closeness
        tolerance = 1e-4
        comparison = tf.abs(x1 - x2) < tolerance

        # Aggregate comparison over all elements: Are all within tolerance?
        all_close = tf.reduce_all(comparison, axis=-1, keepdims=True)  # Shape: (batch_size, 1)

        # Pass one path (e.g., custom dense) through dropout and output layer
        x_out = self.dropout(x1, training=training)
        output_logits = self.out(x_out)

        # Return a dictionary with output and comparison info for debug
        # output_logits: final classification logits
        # all_close: boolean tensor indicating if two dense outputs matched within tolerance per sample

        return {
            "logits": output_logits,
            "dense_outputs_close": all_close,
            "custom_dense_output": x1,
            "reference_dense_output": x2,
        }


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random float32 tensor of shape (batch_size=32, 784)
    # Typical normalized flattened MNIST image shape
    batch_size = 32
    H = 28
    W = 28
    C = 1  # grayscale images, but flattened so C=1 is just for clarity
    shape = (batch_size, H*W)  # (32, 784)
    return tf.random.uniform(shape, dtype=tf.float32)

