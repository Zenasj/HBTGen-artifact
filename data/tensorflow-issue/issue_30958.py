import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.experimental import SequenceFeatures

print('Using Tensorflow version {} (git version {})'.format(tf.version.VERSION, tf.version.GIT_VERSION))

class Toy(Model):

    def __init__(self,
                 fc_list,
                 nb_features,
                 name='toy_model',
                 **kwargs):
        super(Toy, self).__init__(name=name, **kwargs)
        self.fc_list = fc_list
        self.dict_layers = {}
        for fc in self.fc_list:
            fc_name = fc.name
            self.dict_layers[fc_name] = SequenceFeatures(fc)
        self.lstm = LSTM(64, return_sequences=False)
        self.output_layer = Dense(nb_features, activation='softmax')
        
    def call(self, inputs, training=None):
        dict_apply_layers = {}
        for fc in self.fc_list:
            fc_name = fc.name
            if type(fc) == EmbeddingColumn:
                dict_apply_layers[fc_name] = self.dict_layers[fc_name](inputs)[0]
            else:
                # we need to convert inputs[fc_name] to a sparse tensor, see https://github.com/tensorflow/tensorflow/issues/29879
                zero = tf.constant(0, dtype=tf.float32)
                dense = inputs[fc_name]
                indices = tf.where(tf.not_equal(dense, zero))
                values = tf.gather_nd(dense, indices)
                sparse = tf.SparseTensor(indices, values, dense.shape)
                dict_apply_layers[fc_name] = self.dict_layers[fc_name]({fc_name: sparse})[0]
        x = tf.concat([v for _, v in dict_apply_layers.items()], axis=-1)
        x = self.lstm(x)
        x = self.output_layer(x)
        return x
    
#Dataset Parameters
nb_batches = 15
batch_size = 24
sequence_length = 9
nb_features = 10

#Dataset construction
input_dense = tf.constant(np.random.normal(0, 1, (nb_batches, batch_size, sequence_length)))
input_dense = tf.cast(input_dense, dtype=tf.float32)
input_cat = tf.constant(np.random.randint(0, nb_features, (nb_batches, batch_size, sequence_length)))
input_dict = {'dense': input_dense, 'categorical': input_cat}
input_dataset = tf.data.Dataset.from_tensor_slices(input_dict)

target_cat = tf.constant(np.random.randint(0, high=nb_features, size=(nb_batches, batch_size)))
target_dataset = tf.data.Dataset.from_tensor_slices(target_cat)

training_dataset = tf.data.Dataset.zip((input_dataset, target_dataset))

#Feature columns definition
fc_dense = tf.feature_column.sequence_numeric_column('dense')
fc_cat = tf.feature_column.sequence_categorical_column_with_identity('categorical', nb_features)
embedding_units = 16
fc_cat = tf.feature_column.embedding_column(fc_cat, embedding_units)
    
#Model Parameters
rnn_units = 64

#Training Parameters
epochs = 2

#Try the model with the sequence_numeric_column feature column
model = Toy([fc_dense,], nb_features)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
try:
    model.fit(x=training_dataset, epochs=epochs)
    model.evaluate(x=training_dataset)
except ValueError as e:
    print(e)
try:
    path = 'tmp/test'
    model.save(path)
    new_model = tf.keras.models.load_model(path)
    new_model.evaluate(x=training_dataset)
except ValueError as e:
    print(e)
    
#Try the call method with the sequence_numeric_column feature column
model = Toy([fc_dense], nb_features)
try:
    for x, y in training_dataset:
        model(x)
    print('call worked')
except ValueError as e:
    print(e)

#Try the call method in graph mode with the sequence_numeric_column feature column
@tf.function
def call_graph(model, inputs):
    return model(inputs)
model = Toy([fc_dense], nb_features)
try:
    for x, y in training_dataset:
        call_graph(model, x)
    print('call in graph mode worked')
except ValueError as e:
    print(e)
    
#Try the model with the embedding_column feature column
model = Toy([fc_cat,], nb_features)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
try:
    model.fit(x=training_dataset, epochs=epochs)
    model.evaluate(x=training_dataset)
except ValueError as e:
    print(e)
try:
    path = 'tmp/test'
    model.save(path)
    new_model = tf.keras.models.load_model(path)
    new_model.evaluate(x=training_dataset)
except ValueError as e:
    print(e)

for x, y in training_dataset.take(1):
    model(x)