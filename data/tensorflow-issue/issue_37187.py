import tensorflow as tf
from tensorflow import keras

# Define the custom callback for the metric
class CustomMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        logs['customer_metric_name'] = ... # compute the metric value

# Initialize the checkpoint callback with the custom metric
weights_path = 'checkpoint-{epoch:02d}-{customer_metric_name:.4f}.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_path, monitor='customer_metric_name')
checkpoint._supports_tf_logs = False

# Define the model callbacks
callbacks = [
    CustomMetricCallback(), # should be listed before the other callbacks that will use its logs dictionary
    checkpoint
]
model.fit(train_gen, validation_data=val_gen, callbacks=callbacks,)