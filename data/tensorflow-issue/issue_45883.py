import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Counter(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = self.add_weight('count', dtype=tf.int64, initializer='zeros')

    def update_state(self, *args, **kwargs):
        self.count.assign_add(1)

    def result(self):
        return self.count


tf.random.set_seed(0)
inp = tf.keras.Input((1,))
out = tf.keras.layers.Dense(2, activation='softmax')(inp)
model = tf.keras.Model(inp, out)

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[Counter()])
batch_size = 1
x = tf.zeros((batch_size, 1,), dtype=tf.float32)
y = tf.zeros((batch_size,), dtype=tf.int64)
model.train_on_batch(x, y, reset_metrics=False)
logs = model.test_on_batch(x, y, reset_metrics=True, return_dict=True)
#  logs contains metrics for both both steps
print(f"counter = {logs['counter']}")  # counter = 2