from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import datetime
import os

import pandas as pd
from numpy import reshape

import tensorflow as tf

EPOCHS = 500
BATCH_SIZE = 256
TEST_SET_RATIO = 0.2

LEARNING_RATE = 0.001
DECAY = 3e-5
LOSS_FUNC = 'categorical_crossentropy'
DROPOUT = 0.2
OUTPUT_PATH = "e:\\ml"

RNN_SEQ_LEN = 128  # number of RNN/LSTM sequence features
L_AMOUNT = 2  # number of labels

MIN_ACC_TO_SAVE_MODEL = 0.6


def create_model():
    new_model = tf.keras.models.Sequential()

    # NETWORK INPUT
    new_model.add(tf.keras.layers.LSTM(RNN_SEQ_LEN, input_shape=TR_FEATURES.shape[1:], return_sequences=True))
    new_model.add(tf.keras.layers.Dropout(DROPOUT))
    new_model.add(tf.keras.layers.BatchNormalization())

    new_model.add(tf.keras.layers.LSTM(RNN_SEQ_LEN, return_sequences=True))
    new_model.add(tf.keras.layers.Dropout(DROPOUT / 2))
    new_model.add(tf.keras.layers.BatchNormalization())

    new_model.add(tf.keras.layers.LSTM(RNN_SEQ_LEN))
    new_model.add(tf.keras.layers.Dropout(DROPOUT))
    new_model.add(tf.keras.layers.BatchNormalization())

    # NETWORK OUTPUT
    new_model.add(tf.keras.layers.Dense(L_AMOUNT, activation=tf.keras.activations.softmax))

    opt = tf.keras.optimizers.Adam(LEARNING_RATE, decay=DECAY)
    new_model.compile(optimizer=opt,
                      loss=LOSS_FUNC,
                      metrics=['accuracy'])

    print(new_model.summary())
    return new_model


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, fp, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', **kwargs):
        super().__init__(fp, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        print("\n-------------------------------------------------------------------------------------------------------")
        print(f"epoch: {epoch}, training_acc: {round(float(logs['accuracy']), 4)}, validation_acc: {round(float(logs['val_accuracy']), 4)}")
        print("-------------------------------------------------------------------------------------------------------\n")

        if MIN_ACC_TO_SAVE_MODEL <= logs['accuracy']:
            super().on_epoch_end(epoch, logs)


if __name__ == '__main__':
    data_filename = 'input_data.csv'
    print("Loading data file: %s" % data_filename)
    dataset = pd.read_csv(data_filename, delimiter=',', header=None)
    dataset = dataset.drop(columns=[0, 1, 2, 3, 4, 5, 6]).values  # drop columns with additional information

    test_set_size = int(len(dataset) * TEST_SET_RATIO)
    print("Test set split at: %d" % test_set_size)

    train_data = dataset[:-test_set_size]
    test_data = dataset[-test_set_size:]  # use most recent data for validation (extract before shuffle)

    TR_F = train_data[:, 0:RNN_SEQ_LEN]
    TS_F = test_data[:, 0:RNN_SEQ_LEN]

    TR_L = train_data[:, RNN_SEQ_LEN:RNN_SEQ_LEN + L_AMOUNT]
    TS_L = test_data[:, RNN_SEQ_LEN:RNN_SEQ_LEN + L_AMOUNT]

    TR_FEATURES = reshape(TR_F, (len(TR_F), RNN_SEQ_LEN, 1))
    TS_FEATURES = reshape(TS_F, (len(TS_F), RNN_SEQ_LEN, 1))

    model = create_model()

    TRAINING_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "sscce_%s" % TRAINING_TIMESTAMP
    os.mkdir("%s\\models\\%s" % (OUTPUT_PATH, model_name))
    filepath = "%s\\models\\%s\\%s--{epoch:02d}-{val_accuracy:.3f}.model" % (OUTPUT_PATH, model_name, model_name)
    checkpoint = CustomModelCheckpoint(filepath,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    log_dir = "%s\\logs\\fit\\%s.model" % (OUTPUT_PATH, model_name)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

    model.fit(x=TR_FEATURES,
              y=TR_L,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              shuffle=True,
              validation_data=(TS_FEATURES, TS_L),
              callbacks=[checkpoint, tensorboard])