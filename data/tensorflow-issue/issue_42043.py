import numpy as np
import tensorflow as tf
from tensorflow import keras

class RocCallback(Callback):
    def __init__(self , dataset_val):
        self.x = dataset_val
        self.y =  np.concatenate([np.array(x[1]) for x in list(dataset_val)]).reshape(-1)
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        pred = model.predict(self.x)
        roc_val = roc_auc_score(self.y, pred)
        logs["roc_val"] = roc_val
        print('\n - %s average: %s' % ('roc_val', str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

tf.keras.callbacks.ModelCheckpoint("model.h5", monitor='roc_val', verbose=0, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch')

# TPUs need this extra setting to save to local disk, otherwise, they can only save models to GCS (Google Cloud Storage).
# The setting instructs Tensorflow to retrieve all parameters from the TPU then do the saving from the local VM, not the TPU.
save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
tf.keras.callbacks.ModelCheckpoint("saved_model", monitor='roc_val', verbose=0, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch', options=save_locally)