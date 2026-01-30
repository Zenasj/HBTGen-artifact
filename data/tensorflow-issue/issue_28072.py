import random
from tensorflow import keras
from tensorflow.keras import layers

import time
import numpy as np
import tensorflow as tf

# Define a scenario
IMAGE_SIZE = 320
CHANNELS_BATCH_SIZE = 2048  # channels * batch_size
REPEATS = 200
SKIP = 10        

#Build various operations        
class build_normal_ops(tf.keras.Model):
    def __init__(self, KERNEL_SIZE, channels):
        super(build_normal_ops, self).__init__()
        self.normal = tf.keras.layers.Conv2D(channels,KERNEL_SIZE,padding="same")
        
    def call(self,x):
        out = self.normal(x)
        return out
    
class build_rank_ops(tf.keras.Model):
    def __init__(self, KERNEL_SIZE, channels):
        super(build_rank_ops, self).__init__()
        self.rs1 = tf.keras.layers.Conv2D(channels,(KERNEL_SIZE,1),padding="same")
        self.rs2 = tf.keras.layers.Conv2D(channels,(1,KERNEL_SIZE),padding="same")
        
    def call(self,x):
        out = self.rs1(x)
        out = self.rs2(out)
        return out    
    
class build_depth_ops(tf.keras.Model):
    def __init__(self, KERNEL_SIZE, channels):
        super(build_depth_ops, self).__init__()
        self.depthwise = tf.keras.layers.DepthwiseConv2D(KERNEL_SIZE,padding="same")
        self.pointwise = tf.keras.layers.Conv2D(channels,1,padding="same")
        
    def call(self,x):
        out = self.depthwise(x)
        out = self.pointwise(out)        
        return out
       
def build_ops_all(channels, kernel_size):    
    normal = build_normal_ops(kernel_size,channels)
    rank = build_rank_ops(kernel_size,channels)
    depth = build_depth_ops(kernel_size,channels)    
    return normal, rank, depth 

def time_ops(ops: tf.Operation):
    # Benchmark operation
    with tf.device("GPU"):
        image = tf.random.normal(shape=[batch_size, IMAGE_SIZE, IMAGE_SIZE, channels], dtype=tf.float32)
        for i in range(REPEATS+SKIP):
            if i == SKIP:
                start = time.time() #Don't time initial runs
            _ = ops(image)
        end = time.time()
        chk = np.round((end - start) / REPEATS * 1000,2)
    return chk 

if __name__ == '__main__':
    #Benchmark with various channel sizes
    for channels in [64,128,256,512]:
        # adjust batch_size so gpu doesn't run out of memory
        batch_size = CHANNELS_BATCH_SIZE // channels        

        #Benchmark with various kernel sizes
        for param in [3,5,7]:                
            normal, rank_separable, depth_separable = build_ops_all(channels, param)
            print('Channels:', channels, 'kernel_size:', param)                   
            
            time_normal = time_ops(normal)
            time_rank = time_ops(rank_separable)
            time_depth = time_ops(depth_separable)

            print("Normal method: {}ms \t Rank-separable method: {}ms \t Depth-separable method: {}ms \n".format(time_normal, time_rank, time_depth))
        print('\n')