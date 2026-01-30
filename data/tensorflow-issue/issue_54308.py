from tensorflow import keras
from tensorflow.keras import layers

tensorflow-metal

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from MLUtils import MLUtils
utils = MLUtils()

ds = pd.read_csv('resources/tweets.csv')
ds.drop('Unnamed: 0', axis=1, inplace=True)
label_encoder = LabelEncoder()
ds['label'] = label_encoder.fit_transform(ds['label'])
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
num_labels = len(le_name_mapping)
X_train, X_test, Y_train, Y_test = train_test_split(ds['tweet'], ds['label'], test_size = 0.2, stratify=ds['label'], random_state=42)

bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural Network Layers
l = tf.keras.layers.Dropout(0.1, name='dropout')(outputs['pooled_output'])
l = tf.keras.layers.Dense(num_labels, name='output', activation='softmax')(l)
model = tf.keras.Model(inputs=[text_input], outputs=[l])

epochs = 15
batch_size = 32
steps_per_epoch = X_train.shape[0] // batch_size
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Model training
model.compile(optimizer='adam', loss=loss, metrics=metrics)
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
model.evaluate(X_test, Y_test)
model.save('text_classifier')