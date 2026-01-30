import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np

#switch here to switch between working keras and non-working tf.keras code
do_broken=True 

if do_broken:
    import tensorflow.compat.v2 as tf
    from tensorflow.compat.v2 import keras
    from tensorflow.compat.v2.keras.layers import Dense
    from tensorflow.compat.v2.keras.models import Sequential
    tf.enable_v2_behavior()
else:
    import keras
    from keras.layers import Dense
    from keras.models import Sequential

import threading
from collections import Generator

class mwe_gen(Generator):
    
    def __init__(self,train_data,train_labels,batch_size):
        self.train_data=train_data
        self.train_labels=train_labels
        self.batch_size=batch_size
        self.batch=0
        self.lock=threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def next(self):
        with self.lock:
            batch=self.batch
            batch_size=self.batch_size
            self.batch=self.batch+self.batch_size
            if self.batch>len(self.train_data):
                self.batch=0
        batch_data=self.train_data[batch:batch+batch_size]
        batch_labels=self.train_labels[batch:batch+batch_size]
        return (batch_data,batch_labels)
    
    def send(self,arg):
        return self.next()
    
    def close(self):
        """Raise GeneratorExit inside generator.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
        
train_data=np.random.normal(size=(10,1))
train_labels=np.random.normal(size=(10,1))

gen=mwe_gen(train_data,train_labels,5)

model=Sequential()
model.add(Dense(1,input_shape=(1,)))

model.compile(loss="mse",optimizer="sgd")

model.fit_generator(gen,steps_per_epoch=2)