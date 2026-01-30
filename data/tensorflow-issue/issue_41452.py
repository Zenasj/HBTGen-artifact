import numpy as np
import random

class MyModel():
   def __init__(self):
       pass
   
   def build_model(self):
       inp = Input(shape = self.outputs.shape)
        
       x = Dense(128, activation = 'elu')(inp) # We'll start exploring with one hidden layer
       outp = Dense(1, activation = 'elu')(x)
        
       self.model = Model(inputs = inp, outputs = outp)

       self.model.compile(loss = dice_loss,
                            optimizer = Adam(),
                            metrics = [dice_coef])

   def run_model(self):
      # For the sake of simplicity
      self.outputs = np.random.rand(8,64,12,86,98,1)      

      # labels shape = (64, 12, 86, 98) to get (1,64,12,86,98)
      self.labels = np.expand_dims(self.labels, axis = (0,-1))
      # outputs shape (8,64,12,86,98,1) to -> (1,64,12,86,98,8)
      self.outputs = np.swapaxes(self.outputs, 0,-1)
      
      self.build_model()

      self.model.fit([self.outputs, self.labels],
                                batch_size = 1,
                                epochs = 200)

model = MyModel()
model.run_model()

self.model.fit([np.expand_dims(self.outputs, axis = 0), self.labels],
                                batch_size = 1,
                                epochs = 200)

class MyModel():
   def __init__(self):
       pass
   
   def build_model(self):
       inp = Input(shape = self.outputs.shape)
        
       x = Dense(128, activation = 'elu')(inp) # We'll start exploring with one hidden layer
       outp = Dense(1, activation = 'elu')(x)
        
       self.model = Model(inputs = inp, outputs = outp)

       self.model.compile(loss = dice_loss,
                            optimizer = Adam(),
                            metrics = [dice_coef])

   def run_model(self):
      # For the sake of simplicity
      self.outputs = np.random.rand(8,64,12,86,98,1)    
      self.labels = np.random.rand(64, 12, 86, 98)

      # labels shape = (64, 12, 86, 98) to get (1,64,12,86,98)
      self.labels = np.expand_dims(self.labels, axis = (0,-1))
      # outputs shape (8,64,12,86,98,1) to -> (1,64,12,86,98,8)
      self.outputs = np.swapaxes(self.outputs, 0,-1)
      
      self.build_model()

      self.model.fit([self.outputs, self.labels],
                                batch_size = 1,
                                epochs = 200)

def dice_coef(y_true, y_pred, smooth=1):
    
    y_true_f =  K.cast(K.flatten(y_true), dtype='float32')
    y_pred_f =  K.cast(K.flatten(y_pred), dtype='float32')

    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) * smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score

def dice_loss(y_true, y_pred, smooth=1):
    return -dice_coef(y_true, y_pred)