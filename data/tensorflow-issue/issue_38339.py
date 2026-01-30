import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def create_model_tf12(model_file):
  from tensorflow.contrib.keras.python.keras.models import Sequential
  from tensorflow.contrib.keras.python.keras.layers import Dense, Embedding
  model = Sequential()
  model.add(Embedding(1000, 64, input_length=10))
  model.save(model_file)

def load_mode_tf_nightly(model_file):
  model = tf.keras.models.load_model(model_file)

{
  'class_name': 'Sequential', 
  'config': [
    {
      'class_name': 'Embedding', 
      'config': {
        'name': 'embedding_1', 
        'trainable': True, 
        'batch_input_shape': [None, 10], 
        'dtype': 'int32', 
        'input_dim': 1000, 
        'output_dim': 64, 
        'embeddings_initializer': {'class_name': 'RandomUniform', 'config': {'minval': 0, 'maxval': None, 'seed': None, 'dtype': 'float32'}}, 
        'embeddings_regularizer': None, 
        'activity_regularizer': None, 
        'embeddings_constraint': None, 
        'mask_zero': False, 
        'input_length': 10
      }
    }
  ]
}

{
  'class_name': 'Sequential', 
  'config': {
    'name': 'sequential', 
    'layers': [
       {
         'class_name': 'InputLayer', 
         'config': {
           'batch_input_shape': [None, 10], 
           'dtype': 'float32', 
           'sparse': False, 
           'ragged': False, 
           'name': 'embedding_input'}
       },
       {
         'class_name': 'Embedding', 
         'config': {
           'name': 'embedding', 
           'trainable': True, 
           'batch_input_shape': [None, 10], 
           'dtype': 'float32', 
           'input_dim': 1000, 
           'output_dim': 64, 
           'embeddings_initializer': {'class_name': 'RandomUniform', 'config': {'minval': -0.05, 'maxval': 0.05, 'seed': None}}, 
           'embeddings_regularizer': None, 
           'activity_regularizer': None, 
           'embeddings_constraint': None, 
           'mask_zero': False, 
           'input_length': 10}
       }
    ]
  }
}