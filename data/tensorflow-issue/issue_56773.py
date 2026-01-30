from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

def _parse_function(example):
    _float_feature = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
    feature_description = {
        'f1': _float_feature,
        'f2': _float_feature,
        'f3': _float_feature,
        'f4': _float_feature,
        'f5': _float_feature,
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    samples = tf.io.parse_example(example, feature_description)
    label = samples['label']
    features = tf.stack([
            samples['f1'],
            samples['f2'],
            samples['f3'],
            samples['f4'],
            samples['f5']],
            axis=1)
    return (features, label)

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

batch_size_per_replica = 256
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_filename = './training_data.tfrec'
train_dataset = tf.data.TFRecordDataset([train_filename]
        ).batch(batch_size
        ).map(_parse_function)
val_filename = './val_data.tfrec'
val_dataset = tf.data.TFRecordDataset([val_filename]
        ).batch(batch_size
        ).map(_parse_function)

train_dataset = strategy.experimental_distribute_dataset(train_dataset)
val_dataset = strategy.experimental_distribute_dataset(val_dataset)

with strategy.scope():
    mdl = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),
        tf.keras.layers.Dense(5),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    mdl.compile(tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy())

h = mdl.fit(
        train_dataset, 
        validation_data=val_dataset,
        verbose=0,
        epochs=50,
        batch_size=batch_size,
        )

train_filename = './training_data.tfrec'
train_dataset = tf.data.TFRecordDataset([train_filename]
    ).batch(batch_size
    ).map(_parse_function
    ).repeat()

val_filename = './val_data.tfrec'
val_dataset = tf.data.TFRecordDataset([val_filename]
    ).batch(batch_size
    ).map(_parse_function
    ).repeat()