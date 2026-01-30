from tensorflow.keras import layers

import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

class TrainingTest(test.TestCase):
  def test_dataset_input_tuples(self):
    with self.test_session():
      a = keras.layers.Input(shape=(3,), name='input_a')
      b = keras.layers.Input(shape=(4,), name='input_b')
      x = keras.layers.concatenate([a, b])
      y = keras.layers.Dense(5, name='dense')(x)

      model = keras.Model(inputs=[a, b], outputs=[y])
      model.compile(loss='mse', metrics=['mae'], optimizer='rmsprop')

      inputs_a = np.zeros((10, 3))
      inputs_b = np.zeros((10, 4))
      targets = np.zeros((10, 5))
      dataset = dataset_ops.Dataset.from_tensor_slices(((inputs_a,
                                                         inputs_b),
                                                        targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  def test_distributed_dataset_input_tuples(self):
    with self.test_session():
      a = keras.layers.Input(shape=(3,), name='input_a')
      b = keras.layers.Input(shape=(4,), name='input_b')
      x = keras.layers.concatenate([a, b])
      y = keras.layers.Dense(5, name='dense')(x)
      model = keras.Model(inputs=[a, b], outputs=[y])

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      strategy = mirrored_strategy.MirroredStrategy(['/device:GPU:1',
                                                     '/device:CPU:0'])

      model.compile(loss='mse',
                    metrics=['mae'],
                    optimizer=optimizer,
                    distribute=strategy)

      inputs_a = np.zeros((10, 3))
      inputs_b = np.zeros((10, 4))
      targets = np.zeros((10, 5))
      dataset = dataset_ops.Dataset.from_tensor_slices(((inputs_a,
                                                         inputs_b),
                                                        targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

if __name__ == '__main__':
  test.main()

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class TrainingTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('tuple', lambda *x: x, None),
      ('dict', lambda x, y: {'input_a': x, 'input_b': y}, None),
      ('tuple_distribute',
       lambda *x: x,
       mirrored_strategy.MirroredStrategy(['/device:GPU:1', '/device:CPU:0'])),
      ('dict_distribute',
       lambda x, y: {'input_a': x, 'input_b': y},
       mirrored_strategy.MirroredStrategy(['/device:GPU:1', '/device:CPU:0'])))
  def test_multi_input_model(self, input_fn, distribute):
    with self.test_session():
      a = keras.layers.Input(shape=(3,), name='input_a')
      b = keras.layers.Input(shape=(4,), name='input_b')
      x = keras.layers.concatenate([a, b])
      y = keras.layers.Dense(5, name='dense')(x)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)

      model = keras.Model(inputs=[a, b], outputs=[y])
      model.compile(loss='mse', metrics=['mae'], optimizer=optimizer, distribute=distribute)

      input_a = np.zeros((10, 3))
      input_b = np.zeros((10, 4))
      targets = np.zeros((10, 5))
      dataset = dataset_ops.Dataset.from_tensor_slices((input_fn(input_a, input_b), targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

if __name__ == '__main__':
  test.main()

import numpy as np

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class TrainingTest(test.TestCase):
  def test_multi_input_model(self):
    with self.test_session():
      a = keras.layers.Input(shape=(3,), name='aa_input_a')
      b = keras.layers.Input(shape=(4,), name='zz_input_b')
      x = keras.layers.concatenate([a, b])
      y = keras.layers.Dense(5, name='dense')(x)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      distribute = mirrored_strategy.MirroredStrategy(['/device:GPU:1', '/device:CPU:0'])

      model = keras.Model(inputs=[a, b], outputs=[y])
      model.compile(loss='mse', metrics=['mae'], optimizer=optimizer, distribute=distribute)

      input_a = np.zeros((10, 3))
      input_b = np.zeros((10, 4))
      targets = np.zeros((10, 5))
      dataset = dataset_ops.Dataset.from_tensor_slices(({'aa_input_a': input_a, 'zz_input_b': input_b}, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  def test_multi_input_model_non_alphabetic(self):
    with self.test_session():
      a = keras.layers.Input(shape=(3,), name='zz_input_a')
      b = keras.layers.Input(shape=(4,), name='aa_input_b')
      x = keras.layers.concatenate([a, b])
      y = keras.layers.Dense(5, name='dense')(x)

      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      distribute = mirrored_strategy.MirroredStrategy(['/device:GPU:1', '/device:CPU:0'])

      model = keras.Model(inputs=[a, b], outputs=[y])
      model.compile(loss='mse', metrics=['mae'], optimizer=optimizer, distribute=distribute)

      input_a = np.zeros((10, 3))
      input_b = np.zeros((10, 4))
      targets = np.zeros((10, 5))
      dataset = dataset_ops.Dataset.from_tensor_slices(({'zz_input_a': input_a, 'aa_input_b': input_b}, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

if __name__ == '__main__':
  test.main()