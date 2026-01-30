import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

class PredictPerFeat(object):
   def __init__(self, model, params):

        self.model = tf.keras.models.load_model(model, compile=False)

   def predict(feats):
        start_time = time.time()
        out = self.model.predict_on_batch(feats)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        out = int(out > 0.5) # outputs label

        return out

def main(args):
      predperfeat = PredictPerFeat(args.model_path)
      feats = loadfeats(args.feat_path)   

      for i in range(len(feats)):
         # each feature of shape 1, 25, 40, 1
         # also tested for random input 
         # f = tf.convert_to_tensor(np.random.rand(1,25,40,1))
         f = feats[i]
         pred = predperfeat.predict(feats[i])
         print(prediction)