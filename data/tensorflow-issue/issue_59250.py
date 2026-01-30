from tensorflow.keras import models

#======================================
# accesssing through list like below will result in nearly zero gradient
class Mymodel(keras.models.Model):
   def __init__(self):
       self.layers = [layers.LSTM(hid_size) for i in range(3)]
   def call(self,x):
       for i in range(3):
          x = self.layers[i](x)
       return x