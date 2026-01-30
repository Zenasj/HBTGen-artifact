import tensorflow as tf
from tensorflow import keras

class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs['new_loss']=logs['loss']+logs['val_loss']

save = tf.keras.callbacks.ModelCheckpoint(
        'model.h5', monitor='new_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')

model.fit(
       data, 
        epochs=10, 
        callbacks = [LossCallback(),save], 
        steps_per_epoch=100,
        validation_data=val_data, 
    )

class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LossCallback, self).__init__()
        self._supports_tf_logs = True  ## Adding this fixed my issue

    def on_epoch_end(self, epoch, logs=None):
        logs['new_loss']=logs['loss']+logs['val_loss']