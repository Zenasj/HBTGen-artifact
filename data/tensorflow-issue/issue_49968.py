from tensorflow.keras import models

import sys,os, logging,argparse,math,string,json,gzip
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd 
from tensorflow.keras import layers
import glob,pickle,random
from sklearn.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import *
from tensorflow.python.keras.metrics import sparse_top_k_categorical_accuracy

print(tf.__version__)

from transblock import * 
def acc_top4(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=4)

def acc_top8(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=8)


with open('./app_map_5000.json','r') as f:
    white_5000 = json.load(f)

init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(list(white_5000.keys())),
    values=tf.constant(list([ int(i) for i in white_5000.values()]), dtype=tf.int64))

table = tf.lookup.StaticVocabularyTable(
   init, lookup_key_dtype=tf.string,
   num_oov_buckets=5)

def parser(x):
    x_ = tf.strings.regex_replace(x, '"','')
    tokens = tf.strings.split(x_, sep=',')
    #label = 
    #label = tf.strings.to_number(tokens[-1], tf.int32)
    #features = []
    #words = tf.strings.split(tokens[0], sep=' ')
    #f = table.lookup(words)
    #features.append(tf.concat([f1, tf.expand_dims(f2,-1)], axis=0))
    return tokens[0], table.lookup(tokens[-1])

def get_ds(files, val):
    ds = tf.data.TextLineDataset(files, buffer_size=12800, num_parallel_reads=48 ,compression_type='GZIP')
    ds = ds.map(lambda x: parser(x), tf.data.experimental.AUTOTUNE )
    if not val:
        ds = ds.shuffle(12800)#.repeat()
    ds = ds.batch(128).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


ds_train = get_ds(files=glob.glob('./sents_0420_0428_nofil/part-*.csv.gz'),  val=False)
ds_test =  get_ds(files=glob.glob('./sents_0429_nofil/part-*.csv.gz'),  val=True)

for features in ds_test.take(10):
    print(features)

def get_model_transormer(num_classes):
    preprocessor_file = "./albert_en_preprocess_3" # https://tfhub.dev/tensorflow/albert_en_preprocess/3
    preprocessor_layer = hub.KerasLayer(preprocessor_file)
    preprocessor = hub.load(preprocessor_file)
    vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size'].numpy()

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) 

    encoder_inputs = preprocessor_layer(text_input)['input_word_ids']

    embedding_layer = TokenAndPositionEmbedding(encoder_inputs.shape[1], vocab_size, embed_dim)
    x = embedding_layer(encoder_inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=text_input, outputs=outputs)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["acc", acc_top4, acc_top8])
    return model

# def get_model_bert(num_classes, m='albert'):

#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string
#     m_file = {'albert':"./albert_en_base_2", 'electra':'./electra_base_2', 'dan':"./universal-sentence-encoder_4"}

#     encoder = hub.KerasLayer(m_file[m], trainable=True)

#     if m in ['albert', 'electra']:
#         encoder_inputs = preprocessor_layer(text_input)
#         outputs = encoder(encoder_inputs)
#         embed = outputs["pooled_output"]  
#     elif m in ['dan']:
#         embed = encoder(text_input)
#     else:
#         raise KeyError("model illegal!")

#     out = layers.Dense(num_classes, activation="softmax")(embed)
#     model = tf.keras.Model(inputs=text_input, outputs=out)
#     model.compile(Adam(learning_rate=1e-5), "sparse_categorical_crossentropy", metrics=["acc", acc_top4, acc_top8])
#     return model

# x_train, y_train = df_train['content'].values.reshape(-1,1), df_train['label'].values
# x_test, y_test = df_test['content'].values.reshape(-1,1), df_test['label'].values

#model = get_model_bert(df_train['label'].max()+1)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoint_epoch{epoch}',
    save_weights_only=False,
    monitor='val_acc_top8',
    mode='max',
    save_best_only=False)

model = get_model_transormer(5001)
model.save('checkpoint__')

history = model.fit(
    ds_train, validation_data=ds_test, epochs=1, \
     steps_per_epoch=1000, validation_steps=1000,
     verbose=1, callbacks=[model_checkpoint_callback]
)

# transform to another model which use the dense_2 layer as output
model_file = './checkpoint_epoch'
model = tf.keras.models.load_model(model_file, \
                        custom_objects={"acc_top4":acc_top4, "acc_top8":acc_top8, \
                       "softmax":tf.keras.activations.softmax})
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                 outputs=model.get_layer('dense_2').output)
intermediate_layer_model.save(model_file+"_inter")
y = intermediate_layer_model.predict(ds_test, verbose=1, steps=10)

# save the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_file+"_inter")

#### ver: 2.5.0
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# write the tflite model
tflite_model = converter.convert()
open("{}.tflite".format(model_file+"_inter"), "wb").write(tflite_model)

########## reload 
#model = tf.keras.models.load_model('./model_transoformer_1216_epoch_1')
interpreter = tf.lite.Interpreter(model_path= "{}.tflite".format(model_file+"_inter") )    
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor([sentence]) )
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
pred = output_data[0][0]