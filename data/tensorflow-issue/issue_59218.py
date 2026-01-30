import tensorflow as tf
from tensorflow import keras

class PredictionCallback(tf.keras.callbacks.Callback):    

  def on_epoch_end(self, epoch, logs={}):

    y_pred = self.model.predict(self.X_train)

    print('prediction: {} at epoch: {}'.format(y_pred, epoch))

    pd.DataFrame(y_pred).assign(epoch=epoch).to_csv('{}_{}.csv'.format(filename, epoch))

    cnn_model.fit(X_train, y_train,validation_data=[X_valid,y_valid],epochs=epochs,batch_size=batch_size,
               callbacks=[model_checkpoint,reduce_lr,csv_logger, early_stopping,PredictionCallback()],
               verbose=1)