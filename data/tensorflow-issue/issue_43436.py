import math
import tensorflow as tf

class MyModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)])
    def add(self, x,y):
        return tf.math.add(x,y)