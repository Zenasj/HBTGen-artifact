import numpy as np
import tensorflow as tf
from tensorflow import keras

class DataGenerator(tf.keras.utils.Sequence):
   
    def __init__(self,X,y,batch_size=32):   
    
        self.X = X
        self.y = y
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        L = idx*self.batch_size
        R = L+self.batch_size
        batch_x = self.X[L:R]
        batch_y = self.y[L:R]
        
        return batch_x,batch_y