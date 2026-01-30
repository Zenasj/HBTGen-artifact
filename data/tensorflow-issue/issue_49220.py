from tensorflow import keras

import numpy as np
import math
import re
import pandas as pd
import random

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import nltk
import re
import matplotlib.pyplot as plt

#Configuraciones 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


FullTokenizer = bert.bert_tokenization.FullTokenizer

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

def encode_sentence(sent):
    return ["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"]

data_inputs = [encode_sentence(sentence) for sentence in train["excerpt_lemma"]]

def get_ids(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)

def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)

def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for tok in tokens:
        seg_ids.append(current_seg_id)
        if tok == "[SEP]":
            current_seg_id = 1-current_seg_id # convierte los 1 en 0 y vice versa
    return seg_ids


data_with_len = [[sent, train["target"][i], len(sent)]
                 for i, sent in enumerate(data_inputs)]
# random.shuffle(data_with_len)
# data_with_len.sort(key=lambda x: x[2])
sorted_all = [({"input_1":[get_ids(sent_lab[0]), get_mask(sent_lab[0]), get_segments(sent_lab[0])],
                "input_2":data_train.loc[idx].values},
                sent_lab[1])
              for idx, sent_lab in enumerate(data_with_len)]

all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all,
                                             output_types=({"input_1":tf.int32, "input_2":tf.float32}, tf.float32),
                                             output_shapes=({"input_1":(3, None), "input_2":(18,)}, ()))

BATCH_SIZE = 32
all_batched = all_dataset.padded_batch(BATCH_SIZE,
                                       padded_shapes=({"input_1":(3, None), "input_2":(18,)},()),
                                       padding_values=({"input_1":0, "input_2":0.0}, 0.0))

NB_BATCHES = math.ceil(len(sorted_all) / BATCH_SIZE)
NB_BATCHES_TEST = NB_BATCHES // 10
all_batched.shuffle(NB_BATCHES)

test_dataset = all_batched.take(NB_BATCHES_TEST)
train_dataset = all_batched.skip(NB_BATCHES_TEST)

class DCNNBERTEmbedding(tf.keras.Model):
    
    def __init__(self,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=1,
                 dropout_rate=0.1,
                 name="dcnn"):
        super(DCNNBERTEmbedding, self).__init__(name=name)
        
        self.bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
            trainable=False)

        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,
                                      padding="valid",
                                      activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1,
                                           activation="linear")
    
    def embed_with_bert(self, all_tokens):
        print("HOla")
        _, embs = self.bert_layer([all_tokens["input_1"][:, 0, :],
                                   all_tokens["input_1"][:, 1, :],
                                   all_tokens["input_1"][:, 2, :]])
        return embs

    def call(self, inputs, training):
        #Par
        x = self.embed_with_bert(inputs)

        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output

NB_FILTERS = 20
FFN_UNITS = 20

DROPOUT_RATE = 0.3

NB_EPOCHS = 100

Dcnn = DCNNBERTEmbedding(nb_filters=NB_FILTERS,
                         FFN_units=FFN_UNITS,
                         dropout_rate=DROPOUT_RATE)

Dcnn.compile(loss="mse",
             optimizer="adam",
             metrics=["mse", "mae"])

hist = Dcnn.fit(train_dataset,
         epochs=NB_EPOCHS,
         validation_data = test_dataset)