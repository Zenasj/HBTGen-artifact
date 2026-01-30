import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures([
        tf.feature_column.numeric_column('wf',shape=(47,)),
        tf.feature_column.numeric_column('cf',shape=(5,)),
        tf.feature_column.numeric_column('crl_avg'),
        tf.feature_column.numeric_column('crl_long'),
        tf.feature_column.numeric_column('crl_total'),
    ]),
    
    tf.keras.layers.Dense(40,activation='relu'),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=[tf.keras.metrics.Accuracy()])

model.predict([x_set[0],x_set[1]])