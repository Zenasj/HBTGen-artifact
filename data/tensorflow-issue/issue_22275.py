import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os, sys, logging
import numpy as np
import tensorflow as tf
import itertools as itt
logging.basicConfig(level=logging.INFO)


def _int64_feature(value):
  return tf.train.Feature(int64_list= tf.train.Int64List(value= [value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list= tf.train.BytesList(value= [value]))

def _float32_feature(value):
  return tf.train.Feature(float_list= tf.train.FloatList(value= [value]))

def tf_records_creating(tfrecord_file):
    logging.info('Creating random tfrecord files for 100 sample')

    labels = np.random.uniform(0, num_classes, total_train).astype(np.int32)
    data = np.random.uniform(0, 255, total_train*224*224*3).reshape(total_train, 224, 224, 3).astype(np.int32)

    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    for idx, (image, label) in enumerate(itt.izip(data, labels)):
        image = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'image': _bytes_feature(image),
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return

def decode(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image'], tf.float32)
  image.set_shape([224*224*3])

  image=tf.reshape(image, (224,224,3))

  label = tf.cast(features['label'], tf.int32)
  label_categorical = tf.one_hot(label,depth= num_classes, on_value=1,off_value=0,dtype=tf.int32,)
  label_categorical = tf.reshape(label_categorical, [num_classes])
  label_categorical.set_shape([num_classes])

  return image, label_categorical

def data_preparing(tfrecord_file):

    logging.info('Preparing the Training tf.dataset ')
    training_files = [tfrecord_file]
    dataset_train = tf.data.TFRecordDataset(training_files, num_parallel_reads=1)
    dataset_train = dataset_train.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=4 * batch_size))
    dataset_train = dataset_train.map(decode, num_parallel_calls=1)  
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset_train

def train_model(tfrecord_file):

    base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                                 input_shape=(224, 224, 3), pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False

    logging.info('Building Our Classifier')
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    model.summary()
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    dataset_train = data_preparing(tfrecord_file=tfrecord_file)

    weighted_array_train= np.array([0.01557266,0.00867447,0.04579864,0.08275284,0.18281397,
       0.30659676, 0.04686068, 0.31092999])
    class_weight_dict = dict(enumerate(weighted_array_train))

    if using_class_weight == True:
        model.fit(x=dataset_train, epochs=epochs, verbose=1,class_weight = class_weight_dict,
              steps_per_epoch=int(np.ceil(total_train / batch_size)))
    else:
        model.fit(x=dataset_train, epochs=epochs, verbose=1,
              steps_per_epoch=int(np.ceil(total_train / batch_size)))
    return

if __name__== '__main__':

    total_train = 100.
    num_classes= 8
    batch_size = 10
    epochs = 100

    #TODO (1) :Set the path to tfrecord file that we will create it.
    tfrecord_file = '~/train.tfrecords'
    tf_records_creating(tfrecord_file)                # Implement this only one time

    using_class_weight= False                         # if you set this to True, you will produce the error
    train_model(tfrecord_file)