import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.embed = tf.keras.layers.Embedding(vocab_size, 32)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, input, training=None):
    embedding = self.embed(input)
    token_length = embedding.shape[1]
    print("Token Length:", token_length)
    outputs = tf.TensorArray(tf.float32, token_length)

    for i in tf.range(token_length):
    # for i in range(token_length):
      output = self.dense(embedding[:, i, :])
      outputs = outputs.write(i, output)
    
    return tf.transpose(outputs.stack(), [1,0,2])

with strategy.scope():
  vocab_size = 1000
  batch_size = 32
  sequence_length = 32

  model = TestModel()
  model.compile('adam', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
  model(tf.keras.Input([sequence_length - 1]))

  dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([10000, sequence_length], 0, vocab_size, dtype=tf.int32)).map(lambda x: (x[:-1], x[1:]))
  dataset = dataset.batch(batch_size)

  model.fit(dataset)

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.embed = tf.keras.layers.Embedding(vocab_size, 32)
    self.dense = tf.keras.layers.Dense(vocab_size)
    self.dense2 = tf.keras.layers.Dense(vocab_size)

  def call(self, input, training=None):
    embedding = self.embed(input)
    token_length = embedding.shape[1]
    print("Token Length:", token_length)

    output = self.dense(tf.reduce_mean(embedding, axis=1))
    for i in tf.range(token_length):
      output = self.dense2(output)
    
    return tf.repeat(output[:, tf.newaxis, :], token_length, 1)

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.embed = tf.keras.layers.Embedding(vocab_size, 32)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, input, training=None):
    embedding = self.embed(input)
    token_length = embedding.shape[1]
    print("Token Length:", token_length)

    def cond(i, outputs):
      return i < token_length

    def body(i, outputs):
      output = self.dense(embedding[:, i, :])
      outputs = outputs.write(i, output)
      i += 1
      return i, outputs

    i = 0
    outputs = tf.TensorArray(tf.float32, token_length)
    i, outputs = tf.while_loop(cond, body, [i, outputs], maximum_iterations=token_length)
    
    return tf.transpose(outputs.stack(), [1,0,2])

import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

class TestModel(tf.keras.Model):
  def __init__(self):
    super().__init__()

    self.embed = tf.keras.layers.Embedding(vocab_size, 32)
    self.dense = tf.keras.layers.Dense(vocab_size)

  @tf.function
  def train_step(self, inputs):
    input, target = inputs

    token_length = input.shape[1]
    print("Token Length:", token_length)

    total_loss = 0.0
    with tf.GradientTape() as tape:
      embedding = self.embed(input)

      for i in tf.range(token_length):
        output = self.dense(embedding[:, i, :])
        total_loss += self.compiled_loss(target[:, i], output)
  
    variables = self.trainable_variables 
    gradients = tape.gradient(total_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return {"loss":total_loss}
    

with strategy.scope():
  vocab_size = 1000
  batch_size = 32
  sequence_length = 32

  model = TestModel()
  model.compile('adam', tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

  dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([10000, sequence_length], 0, vocab_size, dtype=tf.int32)).map(lambda x: (x[:-1], x[1:]))
  dataset = dataset.batch(batch_size)

  model.fit(dataset)