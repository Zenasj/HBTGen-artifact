import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
# noinspection PyUnresolvedReferences
from tensorflow_core.python.framework.ops import EagerTensor


def main():
    model = keras.Sequential()
    model.add(layers.Embedding(1000, 64, input_length=10))

if __name__ == '__main__':
    main()

from tensorflow_core.python.framework.ops import EagerTensor
# ...
def write_records(shard_filepath, records):
    with tf.io.TFRecordWriter(shard_filepath) as writer:
        for record in records:
            if isinstance(record, EagerTensor):
                record = record.numpy()
            writer.write(record)

from tensorflow import keras
from tensorflow.keras import layers
# noinspection PyUnresolvedReferences
from tensorflow_core.python.framework.ops import EagerTensor


def main():
    model = keras.Sequential()
    model.add(layers.Embedding(1000, 64, input_length=10))


if __name__ == '__main__':
    main()

from tensorflow.python.framework.ops import EagerTensor