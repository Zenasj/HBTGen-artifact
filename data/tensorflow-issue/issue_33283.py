import random
from tensorflow import keras
from tensorflow.keras import layers

self._thread_local = threading.local()

import pickle
import threading
pickle.dumps(threading.local())

import tensorflow as tf
import dill
import threading
def extractfromlocal(model): #extracts attributes from the local thrading container
    model._thread_local=model._thread_local.__dict__
    for attr in model.__dict__.values():
        if '_thread_local' in dir(attr):
            extractfromlocal(attr)

def loadtolocal(model): #puts attributes back to the local threading container
    aux=threading.local()
    aux.__dict__.update(model._thread_local)
    model._thread_local = aux
    for attr in model.__dict__.values():
        if '_thread_local' in dir(attr):
            loadtolocal(attr)
            
def save_tf_model(model): #saves the model
    extractfromlocal(model)
    with open('mymodel.pkl','wb') as f:
        dill.dump(model,f)
    loadtolocal(model)

def load_tf_model(model):#loads the model
    with open('mymodel.pkl','rb') as f:
        model=dill.load(f)
        loadtolocal(model)
    return model

#just a quick example of this working
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.d = tf.keras.layers.Dense(2)

    def call(self, x):
        return self.d(x)

data=tf.random.normal((2, 3))
model = Model()
print('Before saving',model(data))
save_tf_model(model)
print('After saving',model(data))
model=load_tf_model(model)
print('After loading',model(data))