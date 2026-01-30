from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import pytest
import tensorflow as tf


class FooModel(tf.keras.Model):
    """A basic model for testing.

    Attributes:
        cell: The RNN cell layer.

    """

    def __init__(self, rnn=None, **kwargs):
        """Initialize.

        Args:
            rnn: A Keras RNN layer.
            kwargs:  Additional key-word arguments.

        Raises:
            ValueError: If arguments are invalid.

        """
        super().__init__(**kwargs)

        # Assign layers.
        self.rnn = rnn

    def call(self, inputs, training=None):
        """Call.

        Args:
            inputs: A dictionary of inputs.
            training (optional): Boolean indicating if training mode.

        """
        output = self.rnn(inputs, training=training)
        return output


class BarCell(tf.keras.layers.Layer):
    """RNN cell for testing."""
    def __init__(self, **kwargs):
        """Initialize.

        Args:

        """
        super(BarCell, self).__init__(**kwargs)

        # Satisfy RNNCell contract.
        self.state_size = [tf.TensorShape([1]),]

    def call(self, inputs, states, training=None):
        """Call."""
        output = tf.reduce_sum(inputs, axis=1) + tf.constant(1.0)
        self.add_loss(tf.reduce_sum(inputs))

        states_tplus1 = [states[0] + 1]
        return output, states_tplus1


@pytest.mark.parametrize(
    "is_eager", [True, False]
)
@pytest.mark.parametrize(
    "unroll", [True, False]
)
def test_rnn_fit_with_add_loss(is_eager, unroll):
    """Test fit method (triggering backprop)."""
    tf.config.run_functions_eagerly(is_eager)

    # Some dummy input formatted as a TF Dataset.
    n_example = 5
    x = tf.constant([
        [[1, 2, 3], [2, 0, 0], [3, 0, 0], [4, 3, 4]],
        [[1, 13, 8], [2, 0, 0], [3, 0, 0], [4, 13, 8]],
        [[1, 5, 6], [2, 8, 0], [3, 16, 0], [4, 5, 6]],
        [[1, 5, 12], [2, 14, 15], [3, 17, 18], [4, 5, 6]],
        [[1, 5, 6], [2, 14, 15], [3, 17, 18], [4, 5, 6]],
    ], dtype=tf.float32)
    y = tf.constant(
        [
            [[1], [2], [1], [2]],
            [[10], [2], [1], [7]],
            [[4], [2], [6], [2]],
            [[4], [2], [1], [2]],
            [[4], [2], [1], [2]],
        ], dtype=tf.float32
    )
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(n_example, drop_remainder=False)

    # A minimum model to reproduce the issue.
    cell = BarCell()
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, unroll=unroll)
    model = FooModel(rnn=rnn)
    compile_kwargs = {
        'loss': tf.keras.losses.MeanSquaredError(),
        'optimizer': tf.keras.optimizers.Adam(learning_rate=.001),
    }
    model.compile(**compile_kwargs)

    # Call fit which will trigger gradient computations and raise an error
    # during graph execution.
    model.fit(ds, epochs=1)