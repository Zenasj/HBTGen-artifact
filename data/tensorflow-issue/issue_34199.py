import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


max_seq_len    = 8
channels_count = 11

class MultiOutputModel(tf.keras.Model):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        self.dense_a = tf.keras.layers.Dense(3)
        self.dense_b = tf.keras.layers.Dense(4)
        
    def call(self, inputs):
        seq = inputs["F"]
        out_a = self.dense_a(seq)
        out_b = self.dense_b(seq)
        return {"A": out_a, "B": out_b}
    
def ds_gen():
    while True:
        inputs  = {"F": tf.random.uniform((max_seq_len, channels_count))}
        outputs = {"A": tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32), 
                   "B": tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)}
        yield inputs, outputs
        
ds = tf.data.Dataset.from_generator(ds_gen, 
                                    output_types=({"F": tf.float32}, 
                                                  {"A": tf.int32, "B":tf.int32}), 
                                    output_shapes=({"F": tf.TensorShape([max_seq_len, channels_count])}, 
                                                   {"A":tf.TensorShape([]), "B":tf.TensorShape([])}))
# check dataset - a (features, labels) tuple
for inp, out in ds.batch(8).take(1):
    for ndx, (name, val) in enumerate(inp.items()):
        print("features {}: {}: {}".format(ndx, name, val.shape), val.dtype)
    for ndx, (name, val) in enumerate(out.items()):
        print("  labels {}: {}: {}".format(ndx, name, val.shape), val.dtype)
    
model = MultiOutputModel()

def features_only(feat, lab):
    return feat

pred = model.predict(ds.map(features_only).batch(8).take(1))