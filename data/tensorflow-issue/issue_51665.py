import numpy as np
import math
import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

BATCH_SIZE = 2 ** 7
NUM_ACTION = 11
STATE_DIM = 1

def _huber_loss(y_true, y_pred, max_grad=1.):
    a = tf.abs(y_true - y_pred)
    less_than_max = 0.5 * tf.square(a)
    greater_than_max = max_grad * (a - 0.5 * max_grad)
    return tf.where(a <= max_grad, x=less_than_max, y=greater_than_max)

def mean_huber_loss(y_true, y_pred):
    return tf.reduce_mean(_huber_loss(y_true, y_pred))

class NonDistributionalModel(keras.Model):
    def __init__(self, inputs, outputs):
        super(NonDistributionalModel, self).__init__(inputs=inputs, outputs=outputs)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.abs_metric = keras.metrics.MeanTensor(name="abs") # Returns a tensor with the same shape of the input tensors

        self.criterion = mean_huber_loss

    @tf.function
    def train_step(self, data):
        states, targets = data

        with tf.GradientTape() as tape:
            logits = self(states, training=True)
            loss = self.compiled_loss(targets, logits)
            # loss = self.criterion(targets, logits)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.abs_metric.update_state(tf.reduce_mean(tf.math.abs(targets - logits), axis=-1))
        
        return {"loss": self.loss_tracker.result(), "abs": self.abs_metric.result()}     

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.abs_metric]   

inputs = keras.Input(shape=(STATE_DIM,))
outputs = keras.layers.Dense(NUM_ACTION)(inputs)
model = NonDistributionalModel(inputs, outputs)

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.1,
                                                                       first_decay_steps=1000)
model.compile(optimizer=Adam(lr_schedule), loss=mean_huber_loss)
# model.compile(optimizer=Adam(lr_schedule))

x = np.random.random((BATCH_SIZE, 1))
y = np.random.random((BATCH_SIZE, NUM_ACTION))
model.fit(x, y, batch_size=BATCH_SIZE, epochs=1)

model.save('model')