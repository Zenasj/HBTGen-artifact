from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
import numexpr

data = np.array(
[
    [
        [1 , 10, ],
        [2 , 11, ],
    ],
    [
        [2 , 11, ],
        [3 , 12, ],
    ]
], dtype=np.float32)
y = np.array([101., 202.], dtype=np.float32)

inputs= tf.keras.layers.Input(
shape=(2, 2),
name='input',
)
model = tf.keras.layers.LSTM(
    units=data.shape[2],
    return_sequences=False,
    return_state=False,
    name='lstm',
)(inputs)
model = tf.keras.layers.Dense(
    units=1,
    name='dense',
)(model)
outputs = model
loss = tf.keras.losses.MSE
model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name='model',
)
model.compile(
    optimizer='rmsprop',
    loss='mse',
    metrics='mse',
)
model.summary()
model.fit(
    x=data,
    y=y,
    batch_size=1,
)

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# writer
options = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    flush_mode=None,
    input_buffer_size=None,
    output_buffer_size=None,
    window_bits=None,
    compression_level=0,
    compression_method=None,
    mem_level=None,
    compression_strategy=None,
)
writer = tf.io.TFRecordWriter(
    path=r'test.tfrecord',
    options=options,
)
# iterate over each row
for i in range(data.shape[0]):
    # set example id
    sample_dict = {
        'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
    }
    features_list = {}
    # iterate over each feature
    for c in range(data[0].shape[1]):
        feature_values = [
            _float_feature(v) for v in data[i][:, c]
        ]
        features_list[str(c)] = tf.train.FeatureList(feature=feature_values)
    # set example
    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=sample_dict),
        feature_lists=tf.train.FeatureLists(feature_list=features_list)
    )
    # write
    writer.write(example.SerializeToString())
writer.close()

# read raw
data_raw = tf.data.TFRecordDataset(
    filenames=[r'test.tfrecord'],
    compression_type='ZLIB',
    buffer_size=10*1024, # 10MB
    num_parallel_reads=numexpr.detect_number_of_cores()-1,
)
# parse real
schema = dict(
    zip(
        [str(s) for s in range(data[0].shape[1])],
        [tf.io.FixedLenSequenceFeature([], dtype=tf.float32)] * data[0].shape[1]
    )
)
def decode_fn(record_bytes):
    context, features = tf.io.parse_single_sequence_example(
        serialized=record_bytes,
        context_features={'index': tf.io.FixedLenFeature([], dtype=tf.int64)},
        sequence_features=schema,
    )
    return features
# read real
for r in data_raw.map(decode_fn):
    print(r, '\n')