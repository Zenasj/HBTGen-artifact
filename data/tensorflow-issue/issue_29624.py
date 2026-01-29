# tf.random.normal((2000, 10), dtype=tf.float32) ← input shape inferred from DataGenerator.__getitem__

import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return 2  # fixed number of batches per epoch
    
    def __getitem__(self, index):
        print(f'\n split: {self.split} generator, index: {index}', flush=True)
        y = np.random.uniform(low=0, high=1, size=(2000, 1))
        y = (y > 0.5).astype(np.int32)
        X = np.random.normal(loc=0, scale=1, size=(2000, 10)).astype(np.float32)
        return X, y

    def on_epoch_end(self):
        print(f'on epoch end: {self.split}', flush=True)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.metric_acc = tf.keras.metrics.BinaryAccuracy()

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)
    
    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, loss=self.loss_fn, metrics=[self.metric_acc], **kwargs)

def my_model_function():
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)
    return model

def GetInput():
    # Return a single batch of input data matching input shape (2000, 10) as float32
    return tf.random.normal((2000, 10), dtype=tf.float32)

