from tensorflow import keras

#!/usr/bin/python

from __future__ import absolute_import, division, print_function, unicode_literals

# import comet_ml in the top of your file
from comet_ml import Experiment
from comet_ml import Optimizer
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Add the following code anywhere in your machine learning file

def main():

    # Import and setup data

    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(URL)

    # Setup training test split

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(f'The training set is of length: {len(train)}')
    batch_size = 32

    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    size_train_ds = get_dataset_length(df_to_dataset(train, batch_size=1, repeat=1))
    size_test_ds = get_dataset_length(df_to_dataset(test, batch_size=1, repeat=1))
    size_validation_ds = get_dataset_length(df_to_dataset(val, batch_size=1, repeat=1))

    print(f' training data length is: {size_train_ds}')
    print(f' test data length is: {size_test_ds}')
    print(f' validation data length is: {size_validation_ds}')
    # Configure experiment and hyperparameters to test with.

    hparams = {"nodes":128, "thal_embedding_cols": 3}

    loss = fit_model(train_ds,  # <-- ERROR GETS GENERATED AROUND HERE.
                     test_ds,
                     val_ds,
                     epochs=100,
                     batch_size=batch_size,
                     number_training_examples=size_train_ds,
                     number_testing_examples=size_test_ds,
                     number_validation_examples=size_validation_ds,
                     hyper_parameters=hparams)


    return 0

def df_to_dataset(dataframe, shuffle=True, repeat=None, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size).repeat(repeat)
    return ds

def get_dataset_length(ds):
    """
    This function will get the number of examples in a tf.data.Dataset. Make
    sure that the batch size for the dataset is set to 1, otherwise this
    function will count the number of batches in the data. 
    """
    return ds.reduce(np.int64(0), lambda x, _: x + 1)

def define_features(hyper_parameters):

    feature_columns = []
    # numeric cols
    for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
        feature_columns.append(feature_column.numeric_column(header))
        

    # bucketized cols
    age = feature_column.numeric_column("age")
    age_buckets = feature_column.bucketized_column(age,
                                                   boundaries=[18, 25, 30, 35,
                                                               40, 45, 50, 55,
                                                               60, 65])
    #feature_columns.append(age_buckets)

    # indicator cols
    thal = feature_column.categorical_column_with_vocabulary_list(
        'thal', ['fixed', 'normal', 'reversible'])
    thal_one_hot = feature_column.indicator_column(thal)
    #feature_columns.append(thal_one_hot)

    # embedding cols
    thal_embedding = feature_column.embedding_column(thal,
                                                     dimension= \
                                                     hyper_parameters['thal_embedding_cols'])
    #feature_columns.append(thal_embedding)

    # crossed cols
    crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    #feature_columns.append(crossed_feature)
    return feature_columns

def build_model(features, hyper_parameters):
    '''
    documentation string
    '''
    feature_layer = tf.keras.layers.DenseFeatures(features)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(hyper_parameters['nodes'], activation='relu'),
        layers.Dense(hyper_parameters['nodes'], activation='relu'),
        layers.Dense(hyper_parameters['nodes'], activation='relu'),
        layers.Dense(hyper_parameters['nodes'], activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def fit_model(
              train_dataset,
              test_dataset,
              validation_dataset,
              epochs=None,
              batch_size=None,
              number_training_examples=None,
              number_testing_examples=None,
              number_validation_examples=None,
              hyper_parameters=None):


    model = build_model(define_features(hyper_parameters), hyper_parameters)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_dataset,
              validation_data = test_dataset,
              epochs = epochs,
              steps_per_epoch = number_training_examples//batch_size,
              validation_steps= number_testing_examples//batch_size,
              callbacks = [tensorboard_callback])

    # score = model.evaluate(x_test, y_test, verbose=0)[1]
    loss, accuracy = model.evaluate(validation_dataset,
                                    steps=number_validation_examples//batch_size,
                                    verbose=0)
    print(loss)
    return loss

if __name__ == "__main__":
    main()