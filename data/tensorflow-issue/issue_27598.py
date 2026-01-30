class LineGenerator(object):
  def get_next_line(self):
    while True:
      out = [[6]]
      yield tf.ragged.constant(out, dtype=tf.int64)

class Dataset(object):
  def __init__(self, generator=LineGenerator()):
    self.next_element = self.build_iterator(generator)

  def build_iterator(self, gen: LineGenerator):
    dataset = tf.data.Dataset.from_generator(gen.get_next_line,output_types = tf.int64)
    #some other code...

class Dataset(object):
  def __init__(self, generator=LineGenerator()):
    self.next_element = self.build_iterator(generator)

  def build_iterator(self, gen: LineGenerator):
    dataset = tf.data.Dataset.from_generator(gen.get_next_line,output_types = tf.int64) #the right way
    iter = dataset.make_one_shot_iterator()
    element = iter.get_next() #this line gives the error

    return element
d = Dataset()

import tensorflow as tf


def ragged_tensor_generator():
    while True:
        yield tf.ragged.constant([[1, 2], [1]], dtype=tf.int32)


ds = tf.data.Dataset.from_generator(
    generator=ragged_tensor_generator,
    output_types=tf.int32,
    output_shapes=(2, None))

iterator = iter(ds)
record = next(iterator)

sequence_features = {'my_feature': tf.io.RaggedFeature(dtype=tf.int64)}
result = tf.io.parse_sequence_example(
    example_batch, sequence_features=sequence_features)