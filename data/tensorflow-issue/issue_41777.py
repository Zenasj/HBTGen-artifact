from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc
metadata = {'m1': tf.ones(shape=(100,1)), 'm2': tf.ones(shape=(100,1)),'label':tf.ones(shape=(100,1))}
num_samples = 100
dnn_optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=1
            )
def meta_dict_gen():
    for i in range(num_samples):
        ls = {}
        for key, val in metadata.items():
            ls[key] = val[i]
        yield ls
# DATASET CREATION
d = tf.data.Dataset.from_generator(
    meta_dict_gen,
    output_types={k: tf.float32 for k in metadata},
    output_shapes={'m1': (1,), 'm2': (1),'label':(1)})
d = d.shuffle(
        buffer_size=10 * 8
    )


features = {'m1':1,'m2':1}
def label_map(d):
    
    label = d.pop('label')
    reshaped_label = tf.reshape(label, [-1, label.shape[-1]])
    reshaped_elem = {
        key: tf.reshape(d[key], [-1, d[key].shape[-1]])
        for key in d if key in features.keys()
    }
    
    return reshaped_elem, reshaped_label
d = d.map(map_func=label_map)

# CREATING DENSE FEATURE LAYER
d_columns = [tf.feature_column.embedding_column(fc.categorical_column_with_hash_bucket(key='m1', hash_bucket_size=2, dtype=tf.int64),dimension=1,combiner='mean'),
tf.feature_column.numeric_column(
                    'm2', shape=(1,))]

d_features = {}
d_features['m1'] = tf.keras.Input(shape=(1,), name='m1', dtype=tf.int64, sparse=False)
d_features['m2'] = tf.keras.Input(shape=(1,), name='m2', dtype=tf.int64, sparse=False)


#CREATING MODEL

d_input = tf.keras.layers.DenseFeatures(d_columns, name='d_embedded')(d_features)
d_output = tf.keras.layers.Dense(1)(d_input)
d_model = tf.keras.Model(d_features,d_output)
d_model.compile()
d_model.compile(optimizer = dnn_optimizer,loss= 'binary_crossentropy',metrics = ['binary_crossentropy'])
d_model.fit(d)