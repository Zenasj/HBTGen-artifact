files = tf.data.Dataset.list_files('/**/regex*')
tensors = files.map(MyCustomReader)

import tensorflow as tf

@tf.function
def CustomReader(path):
    # path must here a string in the current API
    with tf.io.gfile.GFile(name=path, mode='rb') as f:
        
        # Shape information is encoded  on 16 bytes starting at position 40
        f.seek(40)
        shape = tf.io.decode_raw(f.read(16), out_type=tf.int16)
        shape = tf.cast(shape, tf.int32)
       
        # dtype information is encoded on 2 bytes at position 70
        f.seek(70)
        dtype = tf.io.decode_raw(f.read(2), out_type=tf.int16)[0]

        # tensor values populate the  rest of the buffer
        values = tf.io.decode_raw(f.read(-1), out_type=dtype)
        tensor = tf.reshape(values, shape)

        return tensor

class GFileTFRecord:
    def __init__(self, filepath):
        self._gfile_tfrecord = tf.io.gfile.GFile(filepath, 'rb')
    
    def read_proto(self, offset, length):
        self._gfile_tfrecord.seek(offset)
        return self._gfile_tfrecord.read(length)

gfile_tfrecord = GFileTFRecord('file.tfrecord')

dataset = dataset_from_index('file.tfrecord.idx')
dataset = dataset.map(lambda offset, length: gfile_tfrecord.read_proto(offset, length))
dataset = dataset.map(...) # <-- e.g. decode proto