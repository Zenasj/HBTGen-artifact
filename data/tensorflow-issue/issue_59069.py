import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

class AccumOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, optimizer, steps_per_update=1, **kwargs):

        super(AccumOptimizer, self).__init__(name="AccumOptimizer", **kwargs)
        self.optimizer = optimizer
        self.steps_per_update = steps_per_update
        self.iterations = tf.Variable(0, dtype='int64', name='iterations')
        self.cond = tf.equal(self.iterations % self.steps_per_update, 0)
        self.lr = self.optimizer.learning_rate
        self.optimizer.learning_rate = tf.cond(self.cond,
                                               lambda: self.optimizer.learning_rate,
                                               lambda: tf.constant(0, tf.float32))
        ...