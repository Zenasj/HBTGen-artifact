from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import numpy as np

train_epoch_size = 10
test_epoch_size = 20

class KerasBatchGenerator(object):
    def generate(self, phase='train'):
        while True:
            if phase == 'train':
                for i in range(train_epoch_size):
                    yield [np.array([0])],[np.array([0])]
            else:
                for i in range(test_epoch_size):
                    print('   i',i)
                    yield [np.array([0])],[np.array([0])]

keras_gen_train = KerasBatchGenerator()

inputs = Input(shape=(1,))
x = Dense(1)(inputs)

model = Model(inputs=inputs, outputs=x)

model.compile(loss='mean_squared_error', optimizer='SGD')
print(model.summary())

model.fit_generator(keras_gen_train.generate(), train_epoch_size, 2,
                           validation_data=keras_gen_train.generate('test'), validation_steps=test_epoch_size)#, workers = 0)

def _make_enqueued_generator(generator,
                             workers=1,
                             use_multiprocessing=False,
                             max_queue_size=10,
                             shuffle=False):
  """Create a buffered queue of next elements of the generator."""
  is_sequence = isinstance(generator, data_utils.Sequence)
  enqueuer = None
  if workers > 0:
    if is_sequence:
      enqueuer = data_utils.OrderedEnqueuer(
          generator, use_multiprocessing=use_multiprocessing, shuffle=shuffle)
    else:
      enqueuer = data_utils.GeneratorEnqueuer(
          generator, use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers=workers, max_queue_size=max_queue_size) #here the additional calls happen!
    output_generator = enqueuer.get()
  else:
    if is_sequence:
      output_generator = data_utils.iter_sequence_infinite(generator)
    else:
      output_generator = generator
  return output_generator, enqueuer