import numpy as np
import random
import tensorflow as tf

func = load_tflite_model_func(tflite_model_file_path)
runtime_shape = 60, 80
rng = np.random.RandomState(1234)
ims = [rng.randn(*runtime_shape).astype(np.float32) for _ in range(3)]
assert np.allclose(func(ims[0]), ims[0])
assert np.allclose(func(ims[1]), ims[1]-ims[0])
assert np.allclose(func(ims[2]), ims[2]-ims[1])

runtime_shape = 60, 80  # Or any other shape - this shape is not known when the model is saved
func = load_tflite_model_func(tflite_model_file_path, input_shape=runtime_shape)
rng = np.random.RandomState(1234)
ims = [rng.randn(*runtime_shape).astype(np.float32) for _ in range(3)]
assert np.allclose(func(ims[0]), ims[0])
assert np.allclose(func(ims[1]), ims[1]-ims[0])
assert np.allclose(func(ims[2]), ims[2]-ims[1])

class DeltaT:
  def __init__(self): 
    self._last = 0
  def __call__(self, x): 
    delta = x-self._last
    self._last = x 
    return delta 
    
    ...
    func = DeltaT()

class DynamicVariable(tf.Module):
    def __init__(self, max_size: int, dtype: tf.DType, name: Optional[str] = None):
        super().__init__()
        self._max_size = max_size
        self._variable = tf.Variable(tf.zeros(max_size, dtype=dtype), name=name, trainable=False)

    def assign(self, value: tf.Tensor):
        value_flat = tf.reshape(value, [-1])
        assert_op = tf.debugging.assert_less_equal(tf.size(value_flat), self._max_size,
            message=f"Assigned value with size {tf.size(value_flat)} exceeds the maximum allowed size {self._max_size}."
        )
        with tf.control_dependencies([assert_op]):
            self._variable[: tf.size(value)].assign(value_flat)

    def get_value(self, shape: tf.TensorShape) -> tf.Tensor:
        # Add a tf-assertion that the shape fits:
        requested_size = tf.reduce_prod(shape)
        assert_op = tf.debugging.assert_less_equal(requested_size, self._max_size,
            message=f"Requested shape {shape} with size {requested_size} exceeds the maximum allowed size {self._max_size}."
        )
        with tf.control_dependencies([assert_op]):
            return tf.reshape(self._variable[:requested_size], shape)


@dataclass
class TimeDeltaBoundedSize(tf.Module):
    """ Handy for TFLite, which doesn't support dynamic shapes"""
    max_size: int
    _last_val: Optional[DynamicVariable] = None

    def compute_delta(self, arr: tf.Tensor):
        input_shape = tf.shape(arr)
        if self._last_val is None:
            self._last_val = DynamicVariable(self.max_size, dtype=arr.dtype)
        last_val_reshaped = self._last_val.get_value(input_shape)
        delta = arr - last_val_reshaped
        self._last_val.assign(arr)
        return delta