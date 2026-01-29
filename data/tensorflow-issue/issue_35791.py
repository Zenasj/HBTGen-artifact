# tf.random.uniform((BATCH_SIZE, RNN_SEQ_AMOUNT, 1), dtype=tf.float32), tf.random.uniform((BATCH_SIZE, DNN_F_AMOUNT), dtype=tf.float32)
import tensorflow as tf

# Constants inferred from issue details
EPOCHS = 500
BATCH_SIZE = 1000
TEST_SET_RATIO = 0.2

LEARNING_RATE = 0.001
DECAY = 5e-5
LOSS_FUNC = 'categorical_crossentropy'
DROPOUT = 0.2

L_AMOUNT = 2  # number of labels (output classes)
RNN_SEQ_AMOUNT = 128  # length of RNN input sequences
DNN_F_AMOUNT = 4  # number of features for DNN input branch

MIN_ACC_TO_SAVE_MODEL = 0.6  # threshold for saving best model (used in original callback)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # RNN branch layers
        self.lstm1 = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(DROPOUT)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.lstm2 = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(DROPOUT / 2)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.lstm3 = tf.keras.layers.LSTM(RNN_SEQ_AMOUNT)
        self.dropout3 = tf.keras.layers.Dropout(DROPOUT)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()

        # DNN branch layers
        self.dense_dnn1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense_dnn2 = tf.keras.layers.Dense(8, activation='relu')
        self.dense_dnn_out = tf.keras.layers.Dense(L_AMOUNT, activation='softmax')

        # Combined layers
        self.concat = tf.keras.layers.Concatenate()
        self.dense_combined1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_combined2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_out = tf.keras.layers.Dense(L_AMOUNT, activation='softmax')

    def call(self, inputs, training=False):
        # inputs is a tuple/list: (rnn_input, dnn_input)
        rnn_input, dnn_input = inputs

        # RNN branch
        x = self.lstm1(rnn_input)
        x = self.dropout1(x, training=training)
        x = self.batchnorm1(x, training=training)

        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.batchnorm2(x, training=training)

        x = self.lstm3(x)
        x = self.dropout3(x, training=training)
        x = self.batchnorm3(x, training=training)

        x = self.flatten(x)

        # DNN branch
        y = self.dense_dnn1(dnn_input)
        y = self.dense_dnn2(y)
        y = self.dense_dnn_out(y)

        # Concatenate RNN and DNN outputs
        combined = self.concat([x, y])

        combined = self.dense_combined1(combined)
        combined = self.dense_combined2(combined)
        output = self.dense_out(combined)

        return output


def my_model_function():
    model = MyModel()
    # Compile model similarly as original create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=DECAY)
    model.compile(optimizer=optimizer,
                  loss=LOSS_FUNC,
                  metrics=['accuracy'])
    return model


def GetInput():
    # Return a tuple of two tensors matching the model's two inputs:
    #   - RNN input shape: (batch_size, RNN_SEQ_AMOUNT, 1)
    #   - DNN input shape: (batch_size, DNN_F_AMOUNT)
    import numpy as np

    rnn_input = tf.random.uniform((BATCH_SIZE, RNN_SEQ_AMOUNT, 1), dtype=tf.float32)
    dnn_input = tf.random.uniform((BATCH_SIZE, DNN_F_AMOUNT), dtype=tf.float32)
    return (rnn_input, dnn_input)

