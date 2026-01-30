import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd

class Adder(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)])# 
  def add(self, x, y):
    return x + y ** 2 + 1
  
  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def square(self, x):
    return x ** 2

to_export = Adder()
tf.saved_model.save(
    to_export, 
    '/tmp/adder'            
)

adder1 = tf.saved_model.load("/tmp/adder")
print(adder1.signatures)
adder1_sig = adder1.signatures["serving_default"]
adder1_sig(x = tf.constant(1.), y = tf.constant(2.))