from tensorflow.keras import Model, layers
import tensorflow as tf

x = layers.Input((10, 10, 1))
t = tf.clip_by_value(x, 0, 1)
pred = layers.Dense(8, activation='relu')(t)
model = Model(inputs=x, outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.save('bla.h5')

from tensorflow.keras import Model, layers
import tensorflow as tf

x = layers.Input((10, 10, 1))
pred = layers.Dense(8, activation='relu')(x)
model = Model(inputs=x, outputs=pred)

def custom_metric(y_pred):
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    return tf.reduce_mean(y_pred)

model.add_metric(custom_metric(pred), name='val_custom', aggregation='mean')

model.compile(loss='binary_crossentropy', optimizer='adam')
model.save('bla.h5')