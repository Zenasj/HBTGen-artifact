from tensorflow import keras
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

batched_features = tf.constant([[[1, 3], [2, 3]],
                                [[2, 1], [1, 2]],
                                [[3, 3], [3, 2]]], shape=(3, 2, 2))
batched_labels = tf.constant([['A', 'A'],
                              ['B', 'B'],
                              ['A', 'B']], shape=(3, 2, 1))
dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
dataset = dataset.batch(1)
for element in dataset.as_numpy_iterator():
  print(element)

class MyTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
    def call(self, inputs, training):
        print(type(inputs))
        feature, lable = inputs
        return feature

model = MyTransformer()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

model.fit(dataset , batch_size = 1, epochs = 1)

model.fit(dataset, batch_size=1, epochs=1)

def call(self, inputs, training):
    feature, lable = inputs
    return feature, lable

batch_size=32
dataset = dataset.batch(batch_size)
model.fit(x=dataset)

class Translator(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units,
               context_text_processor,
               target_text_processor):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(context_text_processor, units)
    decoder = Decoder(target_text_processor, units)

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    context, x = inputs
    context = self.encoder(context)
    logits = self.decoder(context, x)

    #TODO(b/250038731): remove this
    try:
      # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
      del logits._keras_mask
    except AttributeError:
      pass

    return logits