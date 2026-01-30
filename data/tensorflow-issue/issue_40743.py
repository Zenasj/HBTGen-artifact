import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import shutil

import numpy as np
import tensorflow as tf

# build and checkpoint mock pre-trained embeddings
EMBD_INPUT_DIM = 1000
EMBD_OUTPUT_DIM = 64

mock_pretrained_embd = tf.Variable(tf.initializers.GlorotNormal()(shape=(EMBD_INPUT_DIM, EMBD_OUTPUT_DIM)), trainable=True)

ckpt = tf.train.Checkpoint(embeddings=mock_pretrained_embd)
ckpt.write('ckpt/mock_embd_ckpt')

# build keras.Model using EmbeddingColumn in DenseFeatures layer
embd_column = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_identity(key='id', num_buckets=EMBD_INPUT_DIM),
    dimension=EMBD_OUTPUT_DIM,
    ckpt_to_load_from='ckpt/mock_embd_ckpt',
    tensor_name_in_ckpt='embeddings/.ATTRIBUTES/VARIABLE_VALUE',
    trainable=True
)
features_layer = tf.keras.layers.DenseFeatures([embd_column])

x = {'id': tf.keras.Input(shape=(None,), dtype=tf.int64, name='id')}
y = features_layer(x)
y = tf.keras.layers.Dense(1, name='out')(y)
model = tf.keras.Model(inputs=x, outputs=y)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam())

# fit model on mock data to have embeddings updated
TRAIN_BATCH = 64

id_len = np.random.randint(low=0, high=15)
x = {'id': tf.convert_to_tensor(np.random.randint(low=0, high=EMBD_OUTPUT_DIM+1, size=(TRAIN_BATCH, id_len)))}
y = tf.convert_to_tensor(np.random.randint(low=0, high=2, size=(TRAIN_BATCH, 1)), tf.float32)

for i in range(100):
  model.train_on_batch(x=x, y=y)

# export model
model.save('mock_model')

# re-construct keras.Model & SavedModel and invoke on mock data
TEST_INPUT_BATCH = 10
TEST_INPUT_ID_LEN = 5

test_inputs = {'id': tf.random.uniform((TEST_INPUT_BATCH, TEST_INPUT_ID_LEN), minval=0, maxval=EMBD_INPUT_DIM, dtype=tf.int64)}

loaded_keras_model = tf.keras.models.load_model('mock_model')
loaded_keras_model(test_inputs)

loaded_model = tf.saved_model.load('mock_model')
serving_default_fn = loaded_model.signatures['serving_default']
serving_default_fn(**test_inputs)

# remove original embeddings checkpoint and try invoking on mock data again
shutil.rmtree('ckpt')

loaded_keras_model(test_inputs)
serving_default_fn(**test_inputs)