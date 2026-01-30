from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
import random

class BatchGen(tf.keras.utils.Sequence):
    def __len__(self):
        return 100
    
    def __getitem__(self, idx):
        for i in range(100):
            return self.gen_ragged_tensors()
    
    @staticmethod
    def gen_ragged_tensors():
        seq_len = np.random.randint(1,3)
        a = np.zeros((seq_len,2), dtype=np.float32)

        seq_len = np.random.randint(1,3)
        b = np.ones((seq_len,2), dtype=np.float32)
        c = tf.ragged.constant([a,b], dtype=tf.float32)
        return c

class Model(tf.keras.models.Model):
    
    def calc_loss(self, batch_in):
        # dummy operation
        return tf.reduce_mean(batch_in - tf.constant(0, dtype=tf.float32))
    
    @tf.function
    def train_step(self, batch_in):
        
        with tf.GradientTape(persistent=True) as tape:
            prediction_loss = self.calc_loss(batch_in)
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(prediction_gradients, self.trainable_variables))
        self.add_loss(lambda: prediction_loss)
        return {"loss": prediction_loss}
        
    def call(self, inputs, training):
        return self.train_step(inputs)

model = Model()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001))


data_gen = BatchGen()
model.fit(data_gen, epochs=10)

def gen_ragged_tensors():
        seq_len = np.random.randint(1,3)
        a = np.zeros((seq_len,2), dtype=np.float32)
        b = np.ones((seq_len,2), dtype=np.float32)
        c = tf.ragged.constant([a,b], dtype=tf.float32, inner_shape=(2, seq_len, 2))
        return c