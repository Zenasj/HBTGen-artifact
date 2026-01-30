from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import sys

import numpy as np
import tensorflow as tf

USE_ESTIMATOR = True
#USE_ESTIMATOR = False

if USE_ESTIMATOR:
  from tensorflow.python import keras
  from tensorflow.python.keras.models import Sequential
  from tensorflow.python.keras.layers import Dense, LSTM, BatchNormalization
  from tensorflow.python.keras.optimizers import SGD, RMSprop
  N_RECORDS = 1
  BATCH_SIZE = 1
else:
  import keras
  import keras.backend.tensorflow_backend as K
  from keras.models import Sequential
  from keras.layers import Dense, LSTM, BatchNormalization
  from keras.optimizers import SGD, RMSprop
  N_RECORDS = 20
  BATCH_SIZE = 10

#SEQ_LENGTH=1000
#SEQ_LENGTH=5000
#SEQ_LENGTH=10000
SEQ_LENGTH=24000

INPUT_DIM = 598
OUTPUT_DIM = 3

NP_DTYPE = np.float32
TF_DTYPE = tf.float32
  
TRAIN_EPOCHS = 2
DEVICE_ID = '/gpu:0'

class ModelLSTM():
  def __init__(self, batch_size, max_length=None, device_id='/cpu:0', n_input_dim=1, n_output_dim=2):  
    
    self.batch_size = batch_size
    self.max_length = max_length
    self.device_id = device_id
    self.n_input_dim = n_input_dim
    self.n_output_dim = n_output_dim

    self.lstm_n_cell=[100, 100, 100] 
    self.dropout=0.1 
    self.recurrent_dropout=0.1
    
    self.create_model()
        
  def create_model(self):        
      
    with tf.device(self.device_id):
    
      print('Creating Model')
      model = Sequential()
      model.add(LSTM(self.lstm_n_cell[0],
                      return_sequences=True,
                      stateful=False,
                      kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = self.dropout, 
                      recurrent_dropout = self.recurrent_dropout,
                      batch_input_shape=(self.batch_size, self.max_length, self.n_input_dim)))
      model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                   scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
      model.add(LSTM(self.lstm_n_cell[1],
                     return_sequences=True,
                     stateful=False,
                     kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = self.dropout, 
                      recurrent_dropout = self.recurrent_dropout))
      model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                   scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
      model.add(LSTM(self.lstm_n_cell[2],
                     return_sequences=True,
                     stateful=False,
                     kernel_initializer='he_normal',
                      activation='tanh',
                      dropout = self.dropout, 
                      recurrent_dropout = self.recurrent_dropout))
      model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                   scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
       
      model.add(Dense(self.n_output_dim, kernel_initializer='he_normal',
                                      activation='softmax')) 
      
      print (model.summary())
      
      opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  
      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy'],
                    sample_weight_mode='temporal')
    
    self.model = model
    return self
  
  
class TestKerasAndEstimator():
  def __init__(self):
    self.device_id = DEVICE_ID
      
  def set_device(self, id):
    self.device_id = id
          
  def the_input_fn(self, filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _set_shapes(features, labels):
      features.set_shape([SEQ_LENGTH, 598])
      labels.set_shape([SEQ_LENGTH, 3])
      return features, labels

    def _my_parse_function(filename, label=None):
      
      print('Input File:')
      print(filename)
      
      dec_filename = filename.decode(sys.getdefaultencoding())
      print('Decoded File:')
      print(dec_filename)
      
      features = np.zeros((SEQ_LENGTH, INPUT_DIM), dtype=NP_DTYPE)
      labels = np.zeros((SEQ_LENGTH, OUTPUT_DIM), dtype=NP_DTYPE)

      print('Features:')
      print(features)
      print('Labels:')
      print(labels)
      
      return features, labels 
      
     
    labels = [0]*len(filenames)
    labels = np.array(labels)
    labels = tf.constant(labels)
    labels = tf.cast(labels, TF_DTYPE)
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))  
    
    dataset = dataset.map(
      lambda filename, label: tuple(tf.py_func(
        _my_parse_function, [filename, label], [TF_DTYPE, label.dtype])))
    
    print(dataset)

    dataset = dataset.map(_set_shapes)
    
    print("Dataset point 1:")
    print(dataset)
    
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size)
    print("Dataset point 2:")
    print(dataset)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    print("Dataset point 3:")
    print(dataset)
    dataset = dataset.batch(batch_size)  # Batch size to use
    the_iterator = dataset.make_one_shot_iterator()    
    batch_features, batch_labels = the_iterator.get_next()
    print('Batch features') 
    print(batch_features) 
    print('Batch labels') 
    print(batch_labels) 
    return batch_features, batch_labels
  
  def test_keras_estimator(self, n_records=1, batch_size=1):
    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)        
    run_config = tf.estimator.RunConfig(session_config=sess_config)   
    
    train_model = ModelLSTM(batch_size=batch_size, max_length=SEQ_LENGTH, device_id=self.device_id, 
                            n_input_dim=INPUT_DIM, n_output_dim=OUTPUT_DIM)
    self.estimator = tf.keras.estimator.model_to_estimator(keras_model=train_model.model,
                                                           model_dir='.', config=run_config)
    train_records = list()
    for i in range(0, n_records):
      train_records.append('record_' + str(i))
      
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: 
                                        self.the_input_fn(train_records, perform_shuffle=False, batch_size=batch_size), 
                                        max_steps=TRAIN_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: 
                                        self.the_input_fn(train_records, perform_shuffle=False, batch_size=batch_size))

    tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
    
     
  def test_keras(self, n_records=1, batch_size=1):
    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)        
    K.set_session(tf.Session(config=sess_config))
      
    train_model = ModelLSTM(batch_size=batch_size, max_length=SEQ_LENGTH, device_id=self.device_id, 
                            n_input_dim=INPUT_DIM, n_output_dim=OUTPUT_DIM)
    features = np.zeros((n_records, SEQ_LENGTH, INPUT_DIM), dtype=NP_DTYPE)
    labels = np.zeros((n_records, SEQ_LENGTH, OUTPUT_DIM), dtype=NP_DTYPE)

    train_model.model.fit(x=features, y=labels, batch_size=batch_size, epochs=TRAIN_EPOCHS, verbose=1)
    train_model.model.evaluate(x=features, y=labels, batch_size=batch_size, verbose=1)    
     
     
     
if __name__ == "__main__":
  mt = TestKerasAndEstimator()
  if USE_ESTIMATOR:
      mt.test_keras_estimator(n_records=N_RECORDS, batch_size=BATCH_SIZE)
  else:
      mt.test_keras(n_records=N_RECORDS, batch_size=BATCH_SIZE)

import sys

import numpy as np
import tensorflow as tf

USE_ESTIMATOR = True
#USE_ESTIMATOR = False

#USE_CUDNN_LSTM = True
USE_CUDNN_LSTM = False

if USE_ESTIMATOR:
  from tensorflow.python import keras
  from tensorflow.python.keras.models import Sequential
  from tensorflow.python.keras.layers import Dense, LSTM, BatchNormalization, CuDNNLSTM
  from tensorflow.python.keras.optimizers import SGD, RMSprop
  if USE_CUDNN_LSTM:
    N_RECORDS = 35
    BATCH_SIZE = 35
  else:
    N_RECORDS = 1
    BATCH_SIZE = 1
    
else:
  import keras
  import keras.backend.tensorflow_backend as K
  from keras.models import Sequential
  from keras.layers import Dense, LSTM, BatchNormalization
  from keras.optimizers import SGD, RMSprop
  N_RECORDS = 20
  BATCH_SIZE = 10

#SEQ_LENGTH=1000
#SEQ_LENGTH=5000
#SEQ_LENGTH=10000
SEQ_LENGTH=24000

INPUT_DIM = 598
OUTPUT_DIM = 3

NP_DTYPE = np.float32
TF_DTYPE = tf.float32
  
TRAIN_EPOCHS = 2
DEVICE_ID = '/gpu:0'



class ModelLSTM():
  def __init__(self, batch_size, max_length=None, device_id='/cpu:0', n_input_dim=1, n_output_dim=2):  
    
    self.batch_size = batch_size
    self.max_length = max_length
    self.device_id = device_id
    self.n_input_dim = n_input_dim
    self.n_output_dim = n_output_dim

    self.n_layers = 3
    self.lstm_n_cell=[100, 100, 100] 
    self.dropout=0.1 
    self.recurrent_dropout=0.1
    
    self.create_model()
        
  def create_model(self):        
    with tf.device(self.device_id):
    
      print('Creating Model')
      model = Sequential()
      
      for i_layer in range(0, self.n_layers):
        if USE_CUDNN_LSTM:
          model.add(keras.layers.CuDNNLSTM(self.lstm_n_cell[i_layer],
                          return_sequences=True,
                          stateful=False,
                          kernel_initializer='he_normal',
                          batch_input_shape=(self.batch_size, self.max_length, self.n_input_dim)))
        else:
          model.add(LSTM(self.lstm_n_cell[i_layer],
                          return_sequences=True,
                          stateful=False,
                          kernel_initializer='he_normal',
                          activation='tanh',
                          dropout = self.dropout, 
                          recurrent_dropout = self.recurrent_dropout,
                          batch_input_shape=(self.batch_size, self.max_length, self.n_input_dim)))
          
  
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001, center=True, 
                                     scale=True, beta_initializer='zeros', gamma_initializer='ones', 
                                     moving_mean_initializer='zeros', moving_variance_initializer='ones'))      
       
      model.add(Dense(self.n_output_dim, kernel_initializer='he_normal',
                                      activation='softmax')) 
      
      #print (model.summary())
      opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
  
      model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'],
                    weighted_metrics=['accuracy'],
                    sample_weight_mode='temporal')
    
    self.model = model
    return self
  
  
class TestKerasAndEstimator():
  def __init__(self):
    self.device_id = DEVICE_ID
      
  def set_device(self, id):
    self.device_id = id
          
  def the_input_fn(self, filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _set_shapes(features, labels):
      features.set_shape([SEQ_LENGTH, 598])
      labels.set_shape([SEQ_LENGTH, 3])
      return features, labels

    def _my_parse_function(filename, label=None):
      
      print('Input File:')
      print(filename)
      
      dec_filename = filename.decode(sys.getdefaultencoding())
      print('Decoded File:')
      print(dec_filename)
      
      # stub for testing, but normally read data from file here 
      features = np.zeros((SEQ_LENGTH, INPUT_DIM), dtype=NP_DTYPE)
      labels = np.zeros((SEQ_LENGTH, OUTPUT_DIM), dtype=NP_DTYPE)

      print('Features:')
      print(features)
      print('Labels:')
      print(labels)
      
      return features, labels 
      
     
    labels = [0]*len(filenames)
    labels = np.array(labels)
    labels = tf.constant(labels)
    labels = tf.cast(labels, TF_DTYPE)
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))  
    
    dataset = dataset.map(
      lambda filename, label: tuple(tf.py_func(
        _my_parse_function, [filename, label], [TF_DTYPE, label.dtype])))
    
    print(dataset)

    dataset = dataset.map(_set_shapes)
    
    print("Dataset point 1:")
    print(dataset)
    
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size)
    print("Dataset point 2:")
    print(dataset)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    print("Dataset point 3:")
    print(dataset)
    dataset = dataset.batch(batch_size)  # Batch size to use
    the_iterator = dataset.make_one_shot_iterator()    
    batch_features, batch_labels = the_iterator.get_next()
    print('Batch features') 
    print(batch_features) 
    print('Batch labels') 
    print(batch_labels) 
    return batch_features, batch_labels
  
  def test_keras_estimator(self, n_records=1, batch_size=1):
    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)        
    run_config = tf.estimator.RunConfig(session_config=sess_config)   
    
    train_model = ModelLSTM(batch_size=batch_size, max_length=SEQ_LENGTH, device_id=self.device_id, 
                            n_input_dim=INPUT_DIM, n_output_dim=OUTPUT_DIM)
    
        
    self.estimator = tf.keras.estimator.model_to_estimator(keras_model=train_model.model,
                                                           model_dir='models', config=run_config)
    train_records = list()
    for i in range(0, n_records):
      train_records.append('record_' + str(i))
      
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: 
                                        self.the_input_fn(train_records, perform_shuffle=False, batch_size=batch_size), 
                                        max_steps=TRAIN_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: 
                                        self.the_input_fn(train_records, perform_shuffle=False, batch_size=batch_size))

    tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
    #print (train_model.model.summary())
      
    
     
  def test_keras(self, n_records=1, batch_size=1):
    gpu_options = tf.GPUOptions(allow_growth=True) 
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)        
    K.set_session(tf.Session(config=sess_config))
      
    train_model = ModelLSTM(batch_size=batch_size, max_length=SEQ_LENGTH, device_id=self.device_id, 
                            n_input_dim=INPUT_DIM, n_output_dim=OUTPUT_DIM)
    
    # stub for testing, but normally read data from file here 
    features = np.zeros((n_records, SEQ_LENGTH, INPUT_DIM), dtype=NP_DTYPE)
    labels = np.zeros((n_records, SEQ_LENGTH, OUTPUT_DIM), dtype=NP_DTYPE)

    train_model.model.fit(x=features, y=labels, batch_size=batch_size, epochs=TRAIN_EPOCHS, verbose=1)
    train_model.model.evaluate(x=features, y=labels, batch_size=batch_size, verbose=1)    
     
     
     
if __name__ == "__main__":
  mt = TestKerasAndEstimator()
  if USE_ESTIMATOR:
      mt.test_keras_estimator(n_records=N_RECORDS, batch_size=BATCH_SIZE)
  else:
      mt.test_keras(n_records=N_RECORDS, batch_size=BATCH_SIZE)