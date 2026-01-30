from tensorflow.python.keras import keras_parameterized
import os
from tensorflow.python.platform import test
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.keras import testing_utils
from tensorflow.python import keras

class KerasCallbacksTest(keras_parameterized.TestCase):
  def _get_dummy_resource_for_model_checkpoint_testing(self):

    def get_input_datasets():
      # Simple training input.
      train_input = [[1.]] * 16
      train_label = [[0.]] * 16
      ds = dataset_ops.Dataset.from_tensor_slices((train_input, train_label))
      return ds.batch(8, drop_remainder=True)

    # Very simple bias model to eliminate randomness.
    optimizer = gradient_descent.SGD(0.1)
    model = sequential.Sequential()
    model.add(testing_utils.Bias(input_shape=(1,)))
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    train_ds = get_input_datasets()

    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'checkpoint.epoch{epoch:02d}.h5')

    # The filepath shouldn't exist at the beginning.
    self.assertFalse(os.path.exists(filepath))
    callback = keras.callbacks.ModelCheckpoint(
        filepath=filepath, save_weights_only=True)

    return model, train_ds, callback, filepath

  def test_fit_with_ModelCheckpoint_with_dir_as_h5_filepath(self):
    (model, train_ds, callback,
     filepath) = self._get_dummy_resource_for_model_checkpoint_testing()
    temp_dir = self.get_temp_dir()
    filepath = os.path.join(temp_dir, 'temp.h5')

    self.assertFalse(os.path.exists(filepath))
    os.mkdir(filepath)
    self.assertTrue(os.path.exists(filepath))

    callback = keras.callbacks.ModelCheckpoint(filepath=filepath)
    
    with self.assertRaisesRegexp(IOError, 'Please specify a non-directory '
                                        'filepath for ModelCheckpoint.'):
      model.fit(train_ds, epochs=1, callbacks=[callback])

if __name__ == '__main__':
  test.main()