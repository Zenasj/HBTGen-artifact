from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np

def get_model():
    global max_seq_length
    global batch_size
    input_word_ids = keras.layers.Input(batch_shape=(batch_size, max_seq_length, ), 
                                           dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = keras.layers.Input(batch_shape=(batch_size, max_seq_length, ), 
                                       dtype=tf.int32,
                                       name="input_mask")
    segment_ids = keras.layers.Input(batch_shape=(batch_size, max_seq_length, ), 
                                        dtype=tf.int32,
                                        name="segment_ids")
    albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/1",
                                  trainable=True,
                                  name='albert_layer')
    pooled_output, sequence_output = albert_layer([input_word_ids, input_mask, segment_ids])
    output = keras.layers.Dense(2)(sequence_output)

    model = keras.Model(inputs=(input_word_ids, input_mask, segment_ids),
                        outputs=output)
    print(model.summary())
    return model


batch_size = 4
max_seq_length = 16
model = get_model()

input_ids = 5 * np.ones((4, 16), dtype=np.int32)
input_mask = np.ones((4, 16), dtype=np.int32)
segment_ids = np.zeros((4, 16), dtype=np.int32)

with tf.GradientTape(persistent=True) as tape:
    logits = model({
                  'input_word_ids' : input_ids,
                  'input_mask' : input_mask,
                  'segment_ids' : segment_ids
    })
    print(logits)