# tf.random.uniform((B, H, W), dtype=tf.float32)  ← Assumed input shape (None, None), 2D inputs of floats

import tensorflow as tf
from typing import Optional

class DynamicVariable(tf.Module):
    def __init__(self, max_size: int, dtype: tf.DType, name: Optional[str] = None):
        super().__init__()
        self._max_size = max_size
        # Initialize a variable of max_size flat length, zero initialized, non-trainable
        # This is a hacky workaround for TFLite dynamic state with fixed max size
        self._variable = tf.Variable(tf.zeros([self._max_size], dtype=dtype), name=name, trainable=False)

    def assign(self, value: tf.Tensor):
        # Flatten value, assert it fits into max_size, assign slice
        value_flat = tf.reshape(value, [-1])
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(
                tf.size(value_flat), 
                self._max_size,
                message=f"Assigned value size larger than max_size {self._max_size}"
            )
        ]):
            # Assign slice of _variable with the given flattened value
            self._variable[:tf.size(value_flat)].assign(value_flat)

    def get_value(self, shape: tf.TensorShape) -> tf.Tensor:
        # Assert total size fits, then slice and reshape
        requested_size = tf.reduce_prod(shape)
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(
                requested_size,
                self._max_size,
                message=f"Requested shape size larger than max_size {self._max_size}"
            )
        ]):
            return tf.reshape(self._variable[:requested_size], shape)


class MyModel(tf.keras.Model):
    def __init__(self, max_state_size=10000): 
        """
        max_state_size: the maximum flattened size of input/state we allow.
        This is a workaround because TF Lite cannot handle dynamic state shapes.
        """
        super().__init__()
        self._max_state_size = max_state_size
        self._state_var = None

    def call(self, x):
        # x: Tensor of shape (B,H,W) or (H,W), dtype float32 etc
        input_shape = tf.shape(x)
        batch_size = input_shape[0] if tf.rank(x) == 3 else 1
        flat_size = tf.reduce_prod(input_shape[1:]) if tf.rank(x) == 3 else tf.reduce_prod(input_shape)
        # Initialize DynamicVariable on first call if needed
        if self._state_var is None:
            # Initialize DynamicVariable with max size and dtype as x
            self._state_var = DynamicVariable(self._max_state_size, dtype=x.dtype, name="last_input_state")
            # Initialize state variable with zeros to start since no prior input
            zero_init = tf.zeros([self._max_state_size], dtype=x.dtype)
            self._state_var._variable.assign(zero_init)

        # Get last value reshaped to current input shape
        last_val = self._state_var.get_value(input_shape)

        # Compute temporal difference: current input - last input stored in state
        delta = x - last_val

        # Update state variable with current input, assign flattened
        self._state_var.assign(x)

        return delta


def my_model_function():
    # Return an instance with a large max_state_size (adjust as needed)
    # max_state_size chosen as 10000 for example; user can adjust according to max expected input size
    return MyModel(max_state_size=10000)


def GetInput():
    # Generate a random tensor with shape (1, 60, 80) float32 matching the example runtime_shape in issue
    # Assumed batch dimension 1; if user wants variable batch they can adjust accordingly
    # This input must be compatible with MyModel.call
    runtime_shape = (1, 60, 80)
    return tf.random.uniform(runtime_shape, dtype=tf.float32)

