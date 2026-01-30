import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(...),
        ...
    ])
    model.compile(...)
    model.layers[0].set_weights([embedding_matrix])

effnet = efn.EfficientNetB5(weights='imagenet', include_top=False)

model.load_weights('saved_models/wieghts_ef5.h5',by_name = True)
model.compile(loss='mse', optimizer=RAdam(lr=0.00005), metrics=['mse', 'acc'])