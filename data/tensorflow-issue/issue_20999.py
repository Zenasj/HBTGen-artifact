# tf.random.uniform((B, 10), dtype=tf.float32) ← Input shape inferred from example: (20, 10) batch size arbitrary (B), feature size 10

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense = tf.keras.layers.Dense(5, activation='relu')

    def call(self, inputs):
        return self.dense(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor compatible with MyModel's expected input shape
    # Using batch size = 20 to match example, feature size = 10
    return tf.random.uniform((20, 10), dtype=tf.float32)


# Custom LearningRateScheduler callback adapted to work with optimizers using 'learning_rate' attribute
class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler callback compatible with modern optimizers
    that use 'learning_rate' instead of 'lr' attribute.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        # Get current learning rate value (tensor or variable)
        lr_tensor = self.model.optimizer.learning_rate
        lr_value = K.get_value(lr_tensor)
        try:
            # New API: schedule function can take epoch and current lr
            new_lr = self.schedule(epoch, lr_value)
        except TypeError:
            # Old API: schedule function only takes epoch
            new_lr = self.schedule(epoch)
        if not isinstance(new_lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be a float.')

        # Update the learning rate variable/tensor to new value
        K.set_value(lr_tensor, new_lr)
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1:05d}: LearningRateScheduler setting learning rate to {new_lr}.')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)


# Note:
# The code in the issue highlights that older tf.train optimizers lack the "learning_rate" attribute expected by
# Keras callbacks like LearningRateScheduler. Modern tf.keras.optimizers (from TF 1.13+ and in TF 2.x)
# define `.learning_rate` as a variable or schedule object and work well with this callback.
#
# This code snippet does not implement training or the compatibility logic explicitly,
# but provides the model, the input generator, and the updated compatible LearningRateScheduler callback.
#
# Usage in practice would be:
#
#    model = my_model_function()
#    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
#    lr_schedule = LearningRateScheduler(step_decay_fn, verbose=1)
#    model.fit(GetInput(), epochs=..., callbacks=[lr_schedule])
#
# where step_decay_fn is a user-defined schedule function matching callback specs.

