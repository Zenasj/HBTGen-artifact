import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import datetime
from pandas_datareader import data;
from sklearn.model_selection import train_test_split

#get data
everSince = datetime.date(1980, 1, 1)
today = datetime.date(2020, 11, 16)
df = data.get_data_yahoo("CORN", start=everSince,
        end=today);

# extract only 'Close' column
df_close = df[['Close']]

#Convert to TFDataset
WINDOW_SIZE = 10
dataset = tf.data.Dataset.from_tensor_slices((df_close.values))
d = dataset.window(WINDOW_SIZE, shift=1, drop_remainder=True)
d2 = d.flat_map(lambda window: window.batch(WINDOW_SIZE))
len_ds = 0
for item in d2:
    len_ds +=1
d3 = d2.map(lambda window: (window[:-1], window[-1:]))
d_shuffled = d3.shuffle(buffer_size=len_ds, reshuffle_each_iteration=False)

#Split train/val/test
y_targets = np.array([ target.numpy() for _, target in iter(d_shuffled) ])
X_indices = np.arange(len(y_targets))

y_targets = y_targets.reshape((-1,))

#stratify array-like, default=None If not None, data is split in a stratified
#fashion, using this as the class labels.
X_train_indices, X_val_indices, y_train_targets, y_val_targets = train_test_split(
    X_indices, y_targets, test_size=0.15, stratify=None, random_state=53)

X_test_indices, X_val_indices, y_test_targets, y_val_targets = train_test_split(
    X_val_indices, y_val_targets, test_size=0.5, stratify=None, random_state=53)

def get_selected_dataset(ds, X_indices_np):
    # Make a tensor of type tf.int64 to match the one by Dataset.enumerate().
    X_indices_ts = tf.constant(X_indices_np, dtype=tf.int64)

    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns Ture if True is included in the specified tensor.
        return tf.math.reduce_any(index == X_indices_ts)

    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similter to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds.enumerate().filter(is_index_in).map(drop_index)
    return selected_ds

splitted_train_ds = get_selected_dataset(d_shuffled, X_train_indices)
splitted_val_ds   = get_selected_dataset(d_shuffled, X_val_indices)
splitted_test_ds  = get_selected_dataset(d_shuffled, X_test_indices)


def create_model():
    MODEL_ARCH = [
        tf.keras.layers.GRU(50, return_sequences=True, input_shape=( WINDOW_SIZE-1, 1)),
        tf.keras.layers.GRU(50,),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1, activation='tanh'),
        tf.keras.layers.Lambda(lambda x: x*100)

    ]
    model = tf.keras.models.Sequential(MODEL_ARCH)
    return model


model = create_model()
LR = 1e-3
optimizer = tf.keras.optimizers.Adagrad(lr=LR)
model.compile(loss='mse', optimizer=optimizer)

history = model.fit(splitted_train_ds.batch(32), epochs=5,
    validation_data=splitted_val_ds.batch(32), batch_size=32)
model_err = model.evaluate(splitted_test_ds.batch(1), verbose=2)
y_hat = model.predict(splitted_test_ds.batch(1))
y_hat = y_hat.reshape((-1,))
print("model.evaluate(): ", model_err )
print("tf.keras.losses.MSE: ", tf.keras.losses.MSE(y_test_targets, y_hat).numpy())