import random
from tensorflow import keras
from tensorflow.keras import layers

def call(self, inputs):
    """Implements call() for the layer."""
    unpacked_inputs = tf_utils.unpack_inputs(inputs)
    word_embeddings = unpacked_inputs[0]
    token_type_ids = unpacked_inputs[1]
    input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = word_embeddings
    if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      one_hot_ids = tf.one_hot(
          flat_token_type_ids,
          depth=self.token_type_vocab_size,
          dtype=self.dtype)
      token_type_embeddings = tf.matmul(one_hot_ids, self.type_embeddings)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

    if self.use_position_embeddings:
      position_embeddings = tf.expand_dims(
          tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
          axis=0)

      output += position_embeddings

    output = self.output_layer_norm(output)
    output = self.output_dropout(output)

    return output

if self.use_type_embeddings:
      flat_token_type_ids = tf.reshape(token_type_ids, [-1])
      token_type_embeddings = tf.gather(self.type_embeddings, flat_token_type_embeddings)
      token_type_embeddings = tf.reshape(token_type_embeddings,
                                         [batch_size, seq_length, width])
      output += token_type_embeddings

outputs = EmbeddingPostprocessor(
    use_type_embeddings=True,
    token_type_vocab_size=2,
    use_position_embeddings=False,
    dtype=tf.float32)(input1, input2)

import tensorflow as tf
import numpy as np

from official.nlp.bert_modeling import EmbeddingPostprocessor

size = 100

input1 = tf.keras.layers.Input(shape=(size, size), dtype=tf.float32, name='1')
input2 = tf.keras.layers.Input(shape=(size,), dtype=tf.int32, name='2')
outputs = EmbeddingPostprocessor(
    use_type_embeddings=True,
    token_type_vocab_size=2,
    dtype=tf.float32)(input1, input2)
model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)

example1 = tf.constant(np.random.random_sample(size=(1, size, size)), dtype=tf.float32)
example2 = tf.constant(np.random.randint(2, size=(1, size), dtype=np.int32))
example = {'1': example1, '2': example2}

output1 = model.predict(example)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.reset_all_variables()
interpreter.set_tensor(input_details[0]['index'], example[input_details[0]['name']])
interpreter.set_tensor(input_details[1]['index'], example[input_details[1]['name']])
interpreter.invoke()

output2 = interpreter.get_tensor(output_details[0]['index'])

print(np.sum(np.abs(output1 - output2)))