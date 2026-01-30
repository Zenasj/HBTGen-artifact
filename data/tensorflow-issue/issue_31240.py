from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.feature_column import embedding_column, sequence_categorical_column_with_identity, \
    sequence_numeric_column
from tensorflow.keras import Input, Model
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import Input, Dense


#on my computer I have used a
#custom plot_model instead of tf.keras.utils.plot_model 
#because of a bug, see nota bene below
#from deep.modeltodot import plot_model
from tensorflow.keras.utils import plot_model

print(tf.version.GIT_VERSION, tf.version.VERSION)

#Model preparation
seq_fc_dense = sequence_numeric_column('denseFeat')
seq_layer_dense = SequenceFeatures(seq_fc_dense, name='denseFeatLayer')

nb_cat = 5
seq_fc_cat = sequence_categorical_column_with_identity('catFeat', nb_cat)
seq_fc_cat = embedding_column(seq_fc_cat, 2)
seq_layer_cat = SequenceFeatures(seq_fc_cat, name='catFeatLayer')

input_dense = Input(shape=(None,), name='denseFeat')
input_cat = Input(shape=(None,), name='catFeat', dtype=tf.int32)
# we need to convert input_dense to a sparse tensor, see https://github.com/tensorflow/tensorflow/issues/29879
zero = tf.constant(0, dtype=tf.float32)
indices = tf.where(tf.not_equal(input_dense, zero))
values = tf.gather_nd(input_dense, indices)
sparse = tf.SparseTensor(indices, values, tf.cast(tf.shape(input_dense), dtype=tf.int64))

x_dense = seq_layer_dense({'denseFeat': sparse})[0]
x_cat = seq_layer_cat({'catFeat': input_cat})[0]
x = tf.concat([x_dense, x_cat], -1)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs={'denseFeat': input_dense, 'catFeat': input_cat}, outputs=output)

#model.summary()
plot_model(model)

try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None

import numpy as np
import tensorflow as tf
from tensorflow.feature_column import embedding_column, sequence_categorical_column_with_identity, \
    sequence_numeric_column
from tensorflow.keras import Input, Model
from tensorflow.keras.experimental import SequenceFeatures
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.utils import plot_model

print(tf.version.GIT_VERSION, tf.version.VERSION)

#Model preparation
seq_fc_dense = sequence_numeric_column('denseFeat')
seq_layer_dense = SequenceFeatures(seq_fc_dense, name='denseFeatLayer')

nb_cat = 5
seq_fc_cat = sequence_categorical_column_with_identity('catFeat', nb_cat)
seq_fc_cat = embedding_column(seq_fc_cat, 2)
seq_layer_cat = SequenceFeatures(seq_fc_cat, name='catFeatLayer')

input_dense = Input(shape=(None,), name='denseFeat')
input_cat = Input(shape=(None,), name='catFeat', dtype=tf.int32)
# we need to convert input_dense to a sparse tensor, see https://github.com/tensorflow/tensorflow/issues/29879
def sparse_f(input_dense):
    zero = tf.constant(0, dtype=tf.float32)
    indices = tf.where(tf.not_equal(input_dense, zero))
    values = tf.gather_nd(input_dense, indices)
    sparse = tf.SparseTensor(indices, values, tf.cast(tf.shape(input_dense), dtype=tf.int64))
    return sparse
sparse = tf.keras.layers.Lambda(sparse_f)(input_dense)

x_dense = seq_layer_dense({'denseFeat': sparse})[0]
x_cat = seq_layer_cat({'catFeat': input_cat})[0]
x = tf.concat([x_dense, x_cat], -1)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs={'denseFeat': input_dense, 'catFeat': input_cat}, outputs=output)

model.summary()
plot_model(model)