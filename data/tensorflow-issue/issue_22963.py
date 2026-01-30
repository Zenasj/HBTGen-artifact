from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(4)
    
    def call(self, inputs):
  
        return self.dense(inputs)
    
model = MyModel()
# inputs = Input(shape=(None, 10))
model.build((None, 10))
model.summary()