import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
used_compression_type = 'GZIP'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import SGD

#Wrapper
class TFRecordConverter(object):
    def __init__(self):
        pass
    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#Write Sample tfrecords

batch_size = 5
tf_archive = "/tfrecords"
os.makedirs(tf_archive, exist_ok=True)
print(f"TF Records at : {tf_archive}")

def create_test_tfrecords():
    existing_tfrecords = glob.glob(os.path.join(tf_archive,"*"))
    if len(existing_tfrecords) < 1:
        for i in range(0,5):
            x_data = np.random.uniform(low=-1.0, high=5, size= (batch_size*5, 6))
            write_to_TFRecords(os.path.join(tf_archive,f"tffile-{i}.tfr"),x_data)

def create_TFRecordsFile(file):
    Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)    
    compression = tf.io.TFRecordOptions(compression_type = used_compression_type)
    writer =  tf.io.TFRecordWriter(file, options=compression)
    return writer

def write_to_TFRecords(file, values):
    bucket_writer = create_TFRecordsFile(file)
    shape = np.array(values.shape, np.int32)
    example = tf.train.Example(
            features = tf.train.Features(
               feature = {
                    'data':TFRecordConverter().float_feature(values.ravel()), #Float list can only be 1D
                    'shape':TFRecordConverter().bytes_feature(shape.tobytes())
                    }
               ))
    
    print(f"Writing to {file} with size: {shape}")
    bucket_writer.write(example.SerializeToString())
    

def parse_function(serialized_example):       
        
    features = {
        'data': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'shape':tf.io.FixedLenFeature([], tf.string)
        }

    features = tf.io.parse_single_example(serialized=serialized_example, features=features)
    shape = tf.io.decode_raw(features['shape'], tf.int32 )
    dataset = features['data']
    dataset = tf.reshape(dataset, shape)


    return dataset

#Simple dataset generator
def data_gen(batch_size):
    tfrecords = glob.glob(os.path.join(tf_archive,"*"))
    print(f"Loading tfrtecords: {tfrecords}")
    dataset = tf.data.Dataset.from_tensor_slices(tfrecords) 
    dataset = dataset.interleave(lambda x: tf.data.Dataset.from_generator(gen_step, 
                                output_types=( tf.float32, tf.float32), args=(x,)))
        
    dataset = dataset.repeat().batch(batch_size)
    while True:
        for x, Y in dataset:
            yield x, Y

def gen_step(tf_file):
    trRecordDataset = tf.data.TFRecordDataset(tf_file, compression_type=used_compression_type)
    trRecordDataset = trRecordDataset.map(parse_function)
    print(f"Waiting for dataset in step - {tf_file}")
    for dataset in trRecordDataset: 
        Y = dataset[:,0]
        x = dataset[:,1:]
        count = 0
        while count < dataset.shape[0]:
            yield x[count],Y[count]
            count += 1
    print(f"Finsihed dataset in - {tf_file}")

if __name__ == '__main__':
  create_test_tfrecords()
  #trivial model - hangs in 
  model = Sequential()
  model.add(tf.keras.Input(shape=(batch_size,)))
  model.add(Dense(1, activation='linear'))        
  model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics = ['accuracy'])
  
  #model.fit will hang just as it starts to read the second set of tfrecords.
  model.fit(x = data_gen(batch_size=batch_size),
          steps_per_epoch=5,
          epochs=10)
  print(f"Training Finished.")

#working now 
model = Sequential()
model.add(tf.keras.Input(shape=(batch_size,)))
model.add(Dense(1, activation='linear'))        
model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics = ['accuracy'])

 
model.fit(x = data_gen(batch_size=batch_size),
        steps_per_epoch=5,
        epochs=10)
print(f"Training Finished.")