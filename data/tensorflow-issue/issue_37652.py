from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

# generate dummy dataset 
def serialize_example(val, label):
    features = {
      'color': tf.train.Feature(bytes_list=tf.train.BytesList(value=val)),
      'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()

tfrecord_writer = tf.io.TFRecordWriter('./color.tfrecord')
for val, label in [([b'G', b'R'], 1), ([b'B'], 1), ([b'B', b'G'], 0), ([b'R'], 1)]:
    tfrecord_writer.write(serialize_example(val, label))
tfrecord_writer.close()

# load the data generate above
def parse(example_proto):
    feature_description = {
        'color': tf.io.VarLenFeature(tf.string) ,           # ** VarLenFeature **
        'label': tf.io.FixedLenFeature([], tf.int64)        
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    labels = parsed_features.pop('label')
    return parsed_features, labels
    
dataset = tf.data.TFRecordDataset('./color.tfrecord').map(parse).repeat(5).batch(2)

# feature column & inputs.
color_cat = tf.feature_column.categorical_column_with_vocabulary_list(
                    key='color', vocabulary_list=["R", "G", "B"])    

color_emb = tf.feature_column.embedding_column(color_cat, dimension=4, combiner='mean')

inputs = {
    'color': tf.keras.layers.Input(name='color', shape=(None, ), sparse=True, dtype=tf.string)    
}

# build model
deep = tf.keras.layers.DenseFeatures([color_emb, ])(inputs)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(deep)
model = tf.keras.Model(inputs, output)
model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

model.fit(dataset, epochs=5)
model.save('./dummy_model', save_format='tf')