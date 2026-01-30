import tensorflow as tf
from tensorflow import keras

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
      self.step = 1
    
    def on_train_batch_end(self, batch, logs=None):
        if tf.summary.should_record_summaries():
          print(f"Warning!!!!!!!!!!!!: can only record summaries for step: {self.step}")
        self.step += 1

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq=100)
custom_callback = CustomCallback()
model.fit(x=x_train, 
            y=y_train, 
            epochs=epochs, 
            validation_data=(x_test, y_test), 
            callbacks=[tensorboard_callback, custom_callback])