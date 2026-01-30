import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

class Encoder(tf.keras.layers.Layer): 
    def call(self, x, training, mask):
    ##########  it will raise a error when using x.shape[1]  ##########
        seq_len = tf.shape(x)[1]         
        x = self.embedding(x)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v):
  matmul_qk = tf.matmul(q, k, transpose_b=True)
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)
  return output, attention_weights


class TestModel(tf.keras.Model):
  def __init__(self, d_model=128, input_vocab_size=233,target_vocab_size=666):
    super(TestModel, self).__init__()
    self.embedding_inp = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.embedding_tar = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding_inp = tf.random.uniform([1,input_vocab_size,d_model])
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
  
  def call(self, inp, tar):
    ##########      it will raise a error when using inp.shape[1]      ##########
    #####  When use tf.shape or ignore adding self.pos_encoding_inp, it run well  #####
    seq_len = inp.shape[1] # tf.shape(inp)[1]
    inp = self.embedding_inp(inp)
    inp += self.pos_encoding_inp[:, :seq_len, :]

    tar = self.embedding_tar(tar)
    tar, _ = scaled_dot_product_attention(v=inp, k=inp, q=tar)   ## raise error
    out = self.final_layer(tar)
    return out

testmodel = TestModel()

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64),
             tf.TensorSpec(shape=(None, None), dtype=tf.int64)]

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')   

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  with tf.GradientTape() as tape:
    predictions = testmodel(inp, tar_inp)
    loss = loss_object(tar_real, predictions)
  gradients = tape.gradient(loss, testmodel.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, testmodel.trainable_variables))
  print('step over')


input = tf.constant(np.random.randint(0, 100,[64,40]),dtype=tf.int64)
target = tf.constant(np.random.randint(0, 100,[64,39]),dtype=tf.int64)
train_step(input, target)