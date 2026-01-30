from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

@tf.function
def loss(a,b):
    return tf.reduce_mean(tf.abs(a), axis=1)

class FailL(tf.keras.layers.Layer):
    def __init__(self):
        super(FailL, self).__init__()
    
    def call(self, inputs):
        tf.print(tf.ragged.constant([[1],[1,1]]))
        return inputs

class FailM(tf.keras.Model):
    def __init__(self, strategy):
        super(FailM, self).__init__()
        self.strategy = strategy
        if self.strategy is not None:
            with self.strategy.scope():
                self.layer1 = tf.keras.layers.Conv2D(1,[3,3])
                self.layer2 = FailL()
        else:
            self.layer1 = tf.keras.layers.Conv2D(1,[3,3])
            self.layer2 = FailL()
    
    @tf.function
    def call(self, inputs):
        return self.layer2(self.layer1(inputs))
    
    def compile(self):
        if self.strategy is not None:
            with self.strategy.scope():
                super(FailM, self).compile()
                self.loss = loss
                self.optimizer = tf.keras.optimizers.Adam()
        else:
            super(FailM, self).compile()
            self.loss = loss
            self.optimizer = tf.keras.optimizers.Adam()
            
    def train_step(self, data):
        with tf.GradientTape() as tape:
            rag = self.layer2(self.layer1(data))
            loss = self.loss(rag,0)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}
    
    @tf.function
    def distributed_train_step(self, data):
        per_replica_losses = self.strategy.run(self.train_step, args=(data,))
        return {prl: self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[prl], axis=None) for prl in per_replica_losses}
    
    def choose_train_step(self, data):
        if self.strategy is None:
            return self.train_step(data)
        else:
            return self.distributed_train_step(data)

for choose_strat in [None, 
                     tf.distribute.MirroredStrategy(devices=['GPU:0']),
                     ]:
    tf.print('Try with a strategy: ', type(choose_strat))
    model = FailM(choose_strat)
    model.compile()
    res = model.choose_train_step(tf.ones([3,10,10,3]))
    tf.print('Result:', res)