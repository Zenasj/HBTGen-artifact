from tensorflow import keras

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.keras as k

print('TF v:', tf.__version__, 'Keras v:', k.__version__)

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://xx.xx.xx.xx:8470')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)    

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    
    model = k.Sequential()
    model.add(k.layers.Conv1D(filters=16,  kernel_size=2, activation = 'relu', input_shape=(window_size, 1) ))
    model.add(k.layers.Conv1D(filters=32,  kernel_size=2, activation = 'relu'))
    model.add(k.layers.Conv1D(filters=64,  kernel_size=2, activation = 'relu'))
    model.add(k.layers.Conv1D(filters=128, kernel_size=2, activation = 'relu'))
    model.add(k.layers.MaxPooling1D(pool_size=2))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(cats, activation='softmax'))
    
    # summary
    print(model.metrics_names)
    print(model.summary())

    print('--')
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['categorical_accuracy'])
    print('--')

model.fit(X, y, batch_size = window_size, shuffle=False, epochs = 5)