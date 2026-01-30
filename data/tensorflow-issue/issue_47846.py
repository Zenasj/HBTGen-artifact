from tensorflow import keras
from tensorflow.keras import layers

import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text 
import tensorflow_addons as tfa
from official.nlp import optimization
import numpy as np

tf.get_logger().setLevel('ERROR')

bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_preprocess = hub.load(tfhub_handle_preprocess)

def make_bert_preprocess_model(sentence_features, seq_length=128):
  """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

  input_segments = [
      tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
      for ft in sentence_features]

  # Tokenize the text to word pieces.
  bert_preprocess = hub.load(tfhub_handle_preprocess)
  tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
  segments = [tokenizer(s) for s in input_segments]

  truncated_segments = segments

  packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                          arguments=dict(seq_length=seq_length),
                          name='packer')
  model_inputs = packer(truncated_segments)
  return tf.keras.Model(input_segments, model_inputs)

def load_dataset_from_tfds(dataset, batch_size, bert_preprocess_model):

  num_examples = len(list(dataset))

  dataset = dataset.batch(batch_size)
  dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
  dataset = dataset.prefetch(1)
  return dataset, num_examples

sentence_features = ['question', 'answer']
num_classes = 2

bert_preprocess_model = make_bert_preprocess_model(sentence_features)

train_dataset = tf.data.Dataset.from_tensor_slices({'idx':train_df.index.values, 'question':train_df.question.values, 'answer':train_df.answer.values,  'label':train_df.label.values})

val_dataset = tf.data.Dataset.from_tensor_slices({'idx':val_df.index.values, 'question':val_df.question.values, 'answer':val_df.answer.values, 'label':val_df.label.values})

test_dataset = tf.data.Dataset.from_tensor_slices({'idx':test_df.index.values, 'question':test_df.question.values, 'answer':test_df.answer.values, 'label':test_df.label.values})

def build_classifier_model(num_classes, activation=None):
  inputs = dict(
      input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids'),
      input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask'),
      input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids'),
  )

  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='encoder')
  net = encoder(inputs)['pooled_output']
  net = tf.keras.layers.Dropout(rate=0.1)(net)
  net = tf.keras.layers.Dense(num_classes, activation=activation, name='classifier')(net)
  return tf.keras.Model(inputs, net, name='prediction')

strategy = tf.distribute.MirroredStrategy()

batch_size = 32
epochs = 5
init_lr = 2e-5

with strategy.scope():
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

  train_dataset_, train_data_size= load_dataset_from_tfds(train_dataset, batch_size, bert_preprocess_model)
  steps_per_epoch = train_data_size // batch_size
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = num_train_steps // 10

  val_dataset_, val_data_size= load_dataset_from_tfds(val_dataset, batch_size, bert_preprocess_model)
  validation_steps = val_data_size // batch_size

  classifier_model = build_classifier_model(num_classes)


  optimizer = optimization.create_optimizer(init_lr =init_lr , num_train_steps = num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')

  classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

  classifier_model.fit(x = train_dataset_, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data = val_dataset_, validation_steps=validation_steps )