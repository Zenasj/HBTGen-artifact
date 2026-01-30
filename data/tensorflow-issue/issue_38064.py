import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

logs

def doesDataSetContainsEnoughDataForBatch(dataset, batch_size):
    return len(list(dataset.take(batch_size).as_numpy_iterator())) == batch_size
  
def doesDataSetFileContainsEnoughDataForBatch(sampleFileName="", batch_size=100):
    dataset = tf.data.TFRecordDataset(sampleFileName)
    return doesDataSetContainsEnoughDataForBatch(dataset, batch_size=batch_size)

if __name__ == '__main__':
  dataSetFileName="./Samples/validation_data_123.tfrecord"
  if not doesDataSetFileContainsEnoughDataForBatch(dataSetFileName,batch_size=100):
    raise Exception(f"Data set file {dataSetFileName} doesn't contain enough data")
   
  # Now open the data set a second time knowing you have enough data 
  # and use it ....
  """
  trainDataset = tf.data.TFRecordDataset(dataSetFileName)
  
  model.fit ( trainDataset ... 
  
  """

import numpy as np
from tensorflow import keras
model = keras.models.Sequential([keras.layers.Dense(1, input_shape=(1,))])
model.compile(loss="mse", optimizer="adam")
model.fit(x=np.ones((1, 1)), y=np.ones((1, 1)), validation_split=0.5)

logistic_regression.fit(train_dense_x, 
          train_label, epochs=100, batch_size=256,
#           validation_data=(val_dense_x, val_label),  
          callbacks=[tbCallBack])