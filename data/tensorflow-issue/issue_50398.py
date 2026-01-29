# tf.random.uniform((B, input_dim), dtype=tf.float32) ← Assuming input is 2D tensor with shape (batch_size, input_dim)

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops, nn_ops

class DropConnectDense(Dense):
    def __init__(self, *args, rate=0.5, **kwargs):
        self.rate = rate
        if 0. < self.rate < 1.:
            self.uses_learning_phase = True
        else:
            self.uses_learning_phase = False

        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, inputs, training=None):
        # DropConnect applied to weights (kernel and bias) during training phase only.
        def dropped_weights():
            dropped_kernel = nn_ops.dropout(self.kernel, rate=self.rate)
            if self.use_bias:
                dropped_bias = nn_ops.dropout(self.bias, rate=self.rate)
            else:
                dropped_bias = self.bias  # no bias case
            return dropped_kernel, dropped_bias

        def retained_weights():
            return self.kernel, self.bias

        kernel, bias = tf.cond(
            tf.cast(training, tf.bool) if training is not None else tf.constant(False),
            dropped_weights,
            retained_weights
        )

        outputs = gen_math_ops.MatMul(a=inputs, b=kernel)
        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(DropConnectDense, self).get_config()
        config.update({
            'rate': self.rate,
            'uses_learning_phase': self.uses_learning_phase,
        })
        return config


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # For demonstration, assume input_dim=32 and output units=16.
        # You can modify these as needed.
        self.dropconnect_dense = DropConnectDense(16, activation='relu', rate=0.5, input_shape=(32,))
        self.dense = Dense(16, activation='relu')

    def call(self, inputs, training=None):
        # Forward through DropConnectDense
        out_dropconnect = self.dropconnect_dense(inputs, training=training)
        # Forward through regular Dense layer for comparison
        out_dense = self.dense(inputs)

        # Compare outputs approximately for demonstration (e.g., L2 norm difference)
        diff = tf.norm(out_dropconnect - out_dense, ord='euclidean', axis=-1)
        # Return both outputs and the diff as output tuple
        return {'dropconnect_output': out_dropconnect,
                'dense_output': out_dense,
                'difference': diff}


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Assumption: input shape is (batch_size, 32) with float32 dtype
    batch_size = 4  # arbitrary batch size
    input_dim = 32
    # Generate random uniform float tensor as input
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

