import tensorflow as tf
from tensorflow import keras

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch):
        lr = self.model.optimizer.lr
        tf.summary.scalar('learning rate', data=lr, step=epoch)