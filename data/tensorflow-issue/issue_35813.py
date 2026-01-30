from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from threading import Thread
import tensorflow as tf

def make_model():
    tf.keras.layers.Input(10)
    
[Thread(target=make_model).start() for _ in range(10)]

from threading import Thread
import tensorflow as tf

class Model(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10)
    
    def call(self, inputs):
        return self.dense(inputs)
    
def make_model():
    Model()
    
[Thread(target=make_model).start() for _ in range(10)]

import threading
lock = threading.Lock()

lock.acquire()
model = tf.keras.model.model_from_json(...)
lock.release()