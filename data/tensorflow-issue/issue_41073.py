from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

py
import tensorflow as tf
from tensorflow import feature_column
tf.compat.v1.disable_eager_execution()

# Define categorical colunm for our text feature, which is preprocessed into sequence of tokens
text_column = feature_column.sequence_categorical_column_with_vocabulary_list(key='text', vocabulary_list=list(['asd', 'asdf']))

max_length = 6
sequence_feature_layer_inputs = {}
sequence_feature_layer_inputs['text'] = tf.keras.Input(
    shape=(max_length,), name='text', dtype=tf.string)

text_embedding = feature_column.shared_embeddings([text_column], dimension=64)

# below is ok to save
# text_embedding = feature_column.embedding_column(text_column, dimension=8)

# Define SequenceFeatures layer to pass feature_columns into Keras model
sequence_feature_layer = tf.keras.experimental.SequenceFeatures(text_embedding)

# note here that SequenceFeatures layer produce tuple of two tensors as output. We need just first to pass next.
sequence_feature_layer_outputs, _ = sequence_feature_layer(
    sequence_feature_layer_inputs)
x = tf.keras.layers.Conv1D(8, 4)(sequence_feature_layer_outputs)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
# This example supposes binary classification, as labels are 0 or 1
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(
    inputs=[v for v in sequence_feature_layer_inputs.values()], outputs=x)

model.summary()

# This example supposes binary classification, as labels are 0 or 1
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              #run_eagerly=True
              )

model.save('model.h5')