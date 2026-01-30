class DataParser(Hyperparameters):

   
    def __init__(self): 
        self.TFRECORDS_FORMAT=Hyperparameters.TFRECORDS_FORMAT
        self.BATCH_SIZE=Hyperparameters.BATCH_SIZE
        self.HEIGHT=Hyperparameters.HEIGHT
        self.WIDTH=Hyperparameters.WIDTH
        
    def readTFRecs(self,dir_name): 
        
        TFRecFiles=tf.constant(tf.io.gfile.listdir(dir_name))
        TFRecFiles=tf.map_fn(lambda name:dir_name+'/'+name,TFRecFiles)
        TFRecDataset=tf.data.TFRecordDataset(TFRecFiles)#.batch(self.BATCH_SIZE).prefetch(1)
        self.dataset_len=tf.data.experimental.cardinality(TFRecDataset).numpy()
        Dataset = TFRecDataset.map(lambda example:tf.io.parse_example(example,self.TFRECORDS_FORMAT))
        return Dataset
    
    @tf.function
    def decode_image(self,entry):
       return tf.image.decode_image(entry['image'],channels=3) #[batch_size,h,w,3]
    
    
    @tf.function
    def makeDataset(self,TFRecDataset):
        Dataset = TFRecDataset.map(lambda entry: self.decode_image(entry))
        return iter(Dataset)



dp=DataParser()  
TFRecDataset=dp.readTFRecs('../input/cassava-tfrecords-512x512')  
Iter=dp.makeDataset(TFRecDataset)  
next(Iter)

#Import required libraries
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import cv2

class Hyperparameters:
    TFRECORDS_FORMAT={'image': tf.io.FixedLenFeature([], tf.string),
                      'image_name': tf.io.FixedLenFeature([], tf.string),
                      'target': tf.io.FixedLenFeature([], tf.int64)}
    BATCH_SIZE=32
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    HEIGHT=224
    WIDTH=224
    WIDTH_FACTOR=0.2
    HEIGHT_FACTOR=0.2
    FILL_MODE='reflect'
    TRAINING=True