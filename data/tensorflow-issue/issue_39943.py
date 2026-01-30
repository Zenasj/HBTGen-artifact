from tensorflow import keras
from tensorflow.keras import layers

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1),
])

# compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model.fit(train_data, epochs=20)

import numpy as np
import tensorflow as tf

####################################################################################################
####################### helper functions ###########################################################
####################################################################################################

# load dataset from disk in to the memory batch by batch
def get_dataset(params, file_path):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=1, # Artificially small to make examples easier to show.
      label_name=params['label'],
      na_value="?",
      num_epochs=1,
      ignore_errors=True)
  return dataset

# create feature columns for model
def generat_feature_coulumns(NUMERIC_FEATURES,CATEGORIES):
    numeric_columns = []
    for feature in NUMERIC_FEATURES:
      num_col = tf.feature_column.numeric_column(key=feature)
      numeric_columns.append(num_col)
    
    categorical_columns = []
    for feature, vocab in CATEGORIES.items():
      cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
      categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    return   numeric_columns,categorical_columns

# helper function
def pack_row(batch, label):
    return batch, label

####################################################################################################
####################### set parameters   ###########################################################
####################################################################################################
    
# set params
params = {
        'label' : 'survived',
        'labels': [0,1],
        'train' : 'dataset/train.csv',
        'test' : 'dataset/eval.csv'  ,
        'model_dir' : 'results/'
        }

# set the numeric and categorical feature,values
NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']
CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
    }


####################################################################################################
##################################### create model #################################################
####################################################################################################
# generate numeric and categorical feature columns
numeric_columns,categorical_columns =  generat_feature_coulumns(NUMERIC_FEATURES,CATEGORIES)
# make mixed feature layer

model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=params['model_dir'],
    linear_feature_columns=[categorical_columns],
    dnn_feature_columns=[numeric_columns],
    dnn_hidden_units=[100, 50],
    n_classes = 2)

####################################################################################################
####################### train the model   ##########################################################
####################################################################################################

test_dataset = get_dataset(params,params['test'])
train_dataset = get_dataset(params,params['train'])

train_data =  train_dataset.map(pack_row).shuffle(500)
test_data = test_dataset.map(pack_row)
  
model.train(lambda: test_data, steps=100)