import random
from tensorflow.keras import layers
from tensorflow.keras import models

#!/usr/bin/env python3                                                                                                     
import os                                                                                                                  
import sys                                                                                                                 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'                                                                                   
                                                                                                                           
import numpy as np                                                                                                         
# Environment setup, using keras vs. tensorflow.keras                                                                      
                                                                                                                           
import tensorflow as tf                                                                                                    
if sys.argv[1] == 'tensorflow':                                                                                            
    import tensorflow.keras as K                                                                                           
    from tensorflow.keras.models import Model                                                                              
    from tensorflow.keras.layers import Dense, Input, Dropout                                                              
elif sys.argv[1] == 'keras':                                                                                               
    import keras as K                                                                                                      
    from keras.models import Model                                                                                         
    from keras.layers import Dense, Input, Dropout                                                                         
else:                                                                                                                      
    sys.exit()                                                                                                             
                                                                                                                           
epochs = int(sys.argv[2])                                                                                                  
                                                                                                                           
                                                                                                                           
# Trackers of the generated data                                                                                           
train_cntX = 0  # updated for each extracted training batch                                                                
train_cntY = 0  # updated for each extracted training batch                                                                
hacked = []  # updated for each extracted validation batch                                                                 
                                                                                                                           
# Reporting the size of hacked before/after each epoch                                                                     
class TrackingCB(K.callbacks.Callback):                                                                                    
    def __init__(self, steps, *args, **kwargs):                                                                            
        super().__init__(*args, **kwargs)                                                                                  
        self.steps = steps                                                                                                 
                                                                                                                           
    def on_epoch_begin(self, epoch, logs={}):                                                                              
        self.start_length = len(hacked)                                                                                    
                                                                                                                           
    def on_epoch_end(self, epoch, logs={}):                                                                                
        for _ in range(self.steps):                                                                                        
            hacked.pop(0)                                                                                                  
        print('start: %d, end: %d' % (self.start_length, len(hacked)))                                                     
        del self.start_length                                                                                              
                                                                                                                           
# Data generators                                                                                                          
def genX(stddev=0.01, update=True):                                                                                        
    while True:                                                                                                            
        for i in range(20):                                                                                                
            if update:                                                                                                     
                global train_cntX                                                                                          
                train_cntX += 1                                                                                            
            yield {'input': np.random.normal(i + 1, stddev, (32, 10))}                                                     
    return                                                                                                                 
                                                                                                                           
def geny(update=True):                                                                                                     
    while True:                                                                                                            
        for i in range(20):                                                                                                
            if update:                                                                                                     
                global train_cntY                                                                                          
                train_cntY += 1                                                                                            
            yield {'output': np.ones((32, 1)) * ((i + 10) % 20)}                                                           
    return                                                                                                                 
                                                                                                                           
def genTrain():                                                                                                            
    for x, y in zip(genX(), geny()):                                                                                       
        yield x, y                                                                                                         
    return                                                                                                                 
                                                                                                                           
def genValid():                                                                                                            
    for x, y in zip(genX(1e-4, update=False), geny(update=False)):                                                         
        data = x, y                                                                                                        
        hacked.append(data)                                                                                                
        yield data                                                                                                         
    return                                                                                                                 
                                                                                                                           
                                                                                                                           
# Model & training                                                                                                         
inp = Input((10,), name='input')                                                                                           
out = Dense(20, activation='relu')(inp)                                                                                    
out = Dense(20, activation='relu')(out)                                                                                    
out = Dropout(0.5)(out)                                                                                                    
out = Dense(1, name='output')(out)                                                                                         
model = Model(inp, out)                                                                                                    
model.compile(loss=K.losses.mean_squared_error,                                                                            
              optimizer=K.optimizers.Adadelta(0.1, decay=0.01))                                                            
                                                                                                                           
model.fit(genTrain(),                                                                                                      
          epochs=epochs, steps_per_epoch=20,                                                                               
          validation_data=genValid(), validation_steps=20,                                                                 
          callbacks=[TrackingCB(20)], verbose=0)                                                                           
                                                                                                                           
                                                                                                                           
print('tensorflow:', tf.__version__)                                                                                       
print('keras:', K.__version__)                                                                                             
print('train generator counts:', train_cntX, train_cntY)                                                                   
print('validation remaining data length:', len(hacked))