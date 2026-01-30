from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import datetime
import os

import pandas as pd
from numpy import reshape

import tensorflow as tf

EPOCHS = 500
BATCH_SIZE = 1000
TEST_SET_RATIO = 0.2

LEARNING_RATE = 0.001
DECAY = 5e-5
LOSS_FUNC = 'categorical_crossentropy'
DROPOUT = 0.2

L_AMOUNT = 2  # number of labels
RNN_SEQ_AMOUNT = 128  # number of RNN/LSTM sequence features
DNN_F_AMOUNT = 4  # number of DNN branch features

MIN_ACC_TO_SAVE_MODEL = 0.6


def create_model():
    # RNN
    rnn_input = tf.keras.layers.Input(shape=(RNN_SEQ_AMOUNT, 1))

    rnn = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT, return_sequences=True)(rnn_input)
    rnn = tf.keras.layers.Dropout(DROPOUT)(rnn)
    rnn = tf.keras.layers.BatchNormalization()(rnn)

    rnn = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT, return_sequences=True)(rnn)
    rnn = tf.keras.layers.Dropout(DROPOUT / 2)(rnn)
    rnn = tf.keras.layers.BatchNormalization()(rnn)

    rnn = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT)(rnn)
    rnn = tf.keras.layers.Dropout(DROPOUT)(rnn)
    rnn = tf.keras.layers.BatchNormalization()(rnn)

    rnn_output = tf.keras.layers.Flatten()(rnn)

    # DNN
    dnn_input = tf.keras.layers.Input(shape=(DNN_F_AMOUNT,))

    dnn = tf.keras.layers.Dense(16, activation=tf.keras.activations.relu)(dnn_input)
    dnn = tf.keras.layers.Dense(8, activation=tf.keras.activations.relu)(dnn)

    dnn_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)(dnn)

    # Concatenate above outputs
    combined_input = tf.keras.layers.Concatenate()([rnn_output, dnn_output])

    combined_nn = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(combined_input)
    combined_nn = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)(combined_nn)

    combined_output = tf.keras.layers.Dense(units=L_AMOUNT, activation=tf.keras.activations.softmax)(combined_nn)

    # Model setup
    new_model = tf.keras.Model(inputs=[rnn_input, dnn_input], outputs=[combined_output])
    opt = tf.keras.optimizers.Adam(LEARNING_RATE,
                                   decay=DECAY)
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
    data_filename = 'train_2020-01-05_data.csv'
    print("Loading data file: %s" % data_filename)

    dataset = pd.read_csv('e:\\ml\\data\\training\\faulty\\%s' % data_filename, delimiter=',', header=None)
    dataset = dataset.drop(columns=[0, 1, 2, 3, 4, 5, 6]).values  # drop columns with additional information

    test_set_size = int(len(dataset) * TEST_SET_RATIO)
    print("Test set split at: %d" % test_set_size)

    train_data = dataset[:-test_set_size]
    test_data = dataset[-test_set_size:]  # use most recent data for testing

    rnn_tr_f = train_data[:, 0:RNN_SEQ_AMOUNT]
    rnn_ts_f = test_data[:, 0:RNN_SEQ_AMOUNT]

    combined_tr_l = train_data[:, RNN_SEQ_AMOUNT:RNN_SEQ_AMOUNT + L_AMOUNT]
    combined_ts_l = test_data[:, RNN_SEQ_AMOUNT:RNN_SEQ_AMOUNT + L_AMOUNT]

    dnn_tr_f = train_data[:, RNN_SEQ_AMOUNT + L_AMOUNT:]  # enhancement data is added in column after labels
    dnn_ts_f = test_data[:, RNN_SEQ_AMOUNT + L_AMOUNT:]

    rnn_tr_features = reshape(rnn_tr_f, (len(rnn_tr_f), RNN_SEQ_AMOUNT, 1))
    rnn_ts_features = reshape(rnn_ts_f, (len(rnn_ts_f), RNN_SEQ_AMOUNT, 1))

    # create model for combined RNN and DNN
    model = create_model()

    TRAINING_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "faulty_%s" % TRAINING_TIMESTAMP
    os.mkdir("e:\\ml\\models\\%s" % model_name)
    filepath = "e:\\ml\\models\\%s\\%s--{epoch:02d}-{val_accuracy:.3f}.model" % (model_name, model_name)
    checkpoint = CustomModelCheckpoint(filepath,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='max')

    log_dir = "e:\\ml\\logs\\fit\\" + model_name + ".model"
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)

    model.fit(x=[rnn_tr_features, dnn_tr_f],
              y=combined_tr_l,
              epochs=EPOCHS,
              shuffle=True,
              batch_size=BATCH_SIZE,
              validation_data=([rnn_ts_features, dnn_ts_f], combined_ts_l),
              callbacks=[checkpoint, tensorboard])