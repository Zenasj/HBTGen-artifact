import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import test


class TestCase(keras_parameterized.TestCase):
  def test_callback_params_samples(self):
    x, y = np.ones((64, 3)), np.ones((64, 2))
    model = testing_utils.get_small_sequential_mlp(
        num_hidden=10, num_classes=2, input_dim=3)
    model.compile('sgd', 'mse')
    callback = Callback()
    model.evaluate(x, y, callbacks=[callback])
    self.assertEqual(callback.params['samples'], 64)


if __name__ == '__main__':
  test.main()