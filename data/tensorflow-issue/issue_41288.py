from tensorflow.keras import layers
from tensorflow.keras import models

##### IMPORT ANDR PREPARE DATA #######


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np

url = 'https://raw.githubusercontent.com/MislavSag/trademl/master/trademl/modeling/random_forest/X_TEST.csv'
X_TEST = pd.read_csv(url, sep=',')
url = 'https://raw.githubusercontent.com/MislavSag/trademl/master/trademl/modeling/random_forest/labeling_info_TEST.csv'
labeling_info_TEST = pd.read_csv(url, sep=',')


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_TEST.drop(columns=['close_orig']), labeling_info_TEST['bin'],
    test_size=0.10, shuffle=False, stratify=None)


### PREPARE LSTM
x = X_train['close'].values.reshape(-1, 1)
y = y_train.values.reshape(-1, 1)
x_test = X_test['close'].values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
train_val_index_split = 0.75
train_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=int(train_val_index_split*X_TEST.shape[0]),
    shuffle=False,
    reverse=False,
    batch_size=128
)
validation_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x,
    targets=y,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=int((train_val_index_split*X_TEST.shape[0] + 1)),
    end_index=None,  #int(train_test_index_split*X.shape[0])
    shuffle=False,
    reverse=False,
    batch_size=128
)
test_generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data=x_test,
    targets=y_test,
    length=30,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=128
)

# convert generator to inmemory 3D series (if enough RAM)
def generator_to_obj(generator):
    xlist = []
    ylist = []
    for i in range(len(generator)):
        x, y = train_generator[i]
        xlist.append(x)
        ylist.append(y)
    X_train = np.concatenate(xlist, axis=0)
    y_train = np.concatenate(ylist, axis=0)
    return X_train, y_train

X_train_lstm, y_train_lstm = generator_to_obj(train_generator)
X_val_lstm, y_val_lstm = generator_to_obj(validation_generator)
X_test_lstm, y_test_lstm = generator_to_obj(test_generator)

# test for shapes
print('X and y shape train: ', X_train_lstm.shape, y_train_lstm.shape)
print('X and y shape validate: ', X_val_lstm.shape, y_val_lstm.shape)
print('X and y shape test: ', X_test_lstm.shape, y_test_lstm.shape)


##### TRAIN  MODEL #######


model = keras.models.Sequential([
        keras.layers.LSTM(258, return_sequences=True, input_shape=[None, x.shape[1]]),
        
        keras.layers.LSTM(124, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense(1, activation='sigmoid')
        
])
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 
                       keras.metrics.AUC(),
                       keras.metrics.Precision(),
                       keras.metrics.Recall()])
# fit the model
history = model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=128,
                    validation_data=(X_val_lstm, y_val_lstm))



##### SAVE AND LOAD MODEL (WORKS) #######

model.save('my_model_lstm.h5')
model = keras.models.load_model('my_model_lstm.h5')
model.predict(X_test_lstm)

##### SAVE AND LOAD MODEL (DOESNT WORK) #######


model_version = "0001"
model_name = "lstm_cloud"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)

saved_model = tf.saved_model.load(model_path)
y_pred = saved_model(X_test_lstm, training=False)