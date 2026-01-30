import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers

val = pd.read_csv('data/val.csv')
window_length = 40
feats = 4
def get_LSTM_AE_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, feats), return_sequences=True, name='encoder_1'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.RepeatVector(window_length, name='encoder_decoder_bridge'))
    model.add(keras.layers.LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(feats)))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss="mse")
    model.summary()
    
    return model

#distribute training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = 4096
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
with strategy.scope():
    model = get_LSTM_AE_model()

val_events = []
val.groupby('vin').apply(lambda x:val_events.append(x[['a','b','v','d']].values))
def val_data_generator():
    # np.random.shuffle(val_events)
    for events in val_events:
        yield events
val_dataset = tf.data.Dataset.from_generator(
    generator=val_data_generator,
    output_types=tf.float32
)
def tensor_2_window(x):
    x = tf.data.Dataset.from_tensor_slices(x)
    x = x.window(40,shift=1,drop_remainder=True)
    x = x.flat_map(lambda window: window.batch(40))
    return x
val_dataset = val_dataset.flat_map(tensor_2_window)
val_dataset = val_dataset.map(lambda window: (window, window))
val_dataset = val_dataset.cache().batch(4096*9).prefetch(buffer_size=tf.data.AUTOTUNE)
history = model.fit(
    val_dataset,
    epochs=50,
    # validation_data=val_dataset
)

val.groupby('vin').apply(lambda x:val_events.append(x[['a','b','c','d']].values))