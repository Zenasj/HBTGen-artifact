# tf.random.uniform((B, 784), dtype=tf.float32) ← Input shape inferred from MNIST flattened images of shape (batch, 784)

import tensorflow as tf
from tensorflow.keras import layers, Sequential, utils, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn import metrics
import numpy as np

class CustomMetric(Callback):
    def __init__(self, x_valid, y_valid, batch_size=128):
        super().__init__()
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.batch_size = batch_size
        # By default, _supports_tf_logs is False
        # Setting to False means 'logs' is a tf.TensorDict, and 'numpy_logs' contains actual numpy values.
        # This is the source of desync in logs.
        self._supports_tf_logs = False

    def on_epoch_end(self, epoch, logs=None):
        # logs here might be a tensor dict (tf logs), but we set _supports_tf_logs False,
        # so logs passed in is actually numpy logs we can safely modify.
        y_pred = self.model.predict(self.x_valid, batch_size=self.batch_size)
        # Compute log loss between true labels and predicted probs
        val_log_loss = metrics.log_loss(self.y_valid, y_pred)
        logs['val_log_loss'] = val_log_loss


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple MNIST classifier model matching Seq:
        # Input shape: (784,)
        self.dense1 = Dense(64, activation='relu', input_shape=(784,))
        self.dense2 = Dense(32, activation='relu')
        self.dense3 = Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


def my_model_function():
    model = MyModel()
    # Compile model with same config as reported in issue:
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    return model


def GetInput():
    # MNIST inputs flattened and normalized: shape (batch_size, 784)
    # Use batch size 128 as per original code
    batch_size = 128
    input_tensor = tf.random.uniform((batch_size, 784), dtype=tf.float32)
    return input_tensor

