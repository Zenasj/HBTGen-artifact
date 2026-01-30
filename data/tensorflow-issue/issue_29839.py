import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(layers.LSTM(neurons, input_shape=(window_size, inputs_n), return_sequences=True)) 
model.add(layers.LSTM(neurons))
model.add(layers.Dense(outputs_n, activation='sigmoid'))

opt = tf.train.AdamOptimizer(0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
 
tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, 
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(tpu = [TPU_ADDRESS1])))

for epoch in epochs:
    for d in days : 
        # get arrays for the day
        features = np.asarray(d[1])[:,2:9].astype(dtype = 'float32')
        labels = np.asarray(d[1])[:, 9:13].astype(dtype = 'int32')
        
        X,y = split_sequence(features, labels_buy, window_size)

        # train 
        for slide in range(window_size):
            try:
                x1, y1 = X[slide], y[slide]
                x2, y2 = x1.reshape(1,1024,7), y1.reshape(1, 4)
                H = tpu_model.train_on_batch(x2,y2)
            except Exception as e:
                print('** train exception **', e)
                continue

resolver = tf.distribute.cluster_resolver.TPUClusterResolver([TPU_ADDRESS1])
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)

# build model
with tpu_strategy.scope():
    model = tf.keras.Sequential()
    model.add(layers.LSTM(neurons, input_shape=(window_size, inputs_n), return_sequences=True)) 
    model.add(layers.LSTM(neurons))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(outputs_n, activation=activation))

    opt = tf.train.RMSPropOptimizer(learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    X,y = split_sequence(features, labels, window_size)
    print('X shape:', X.shape, 'Y shape:', y.shape)
    for slide in range(window_size):
        x1, y1 = X[slide], y[slide]
        x2, y2 = x1.reshape(1,window_size,inputs_n), y1.reshape(1,outputs_n)
        model.train_on_batch(x2,y2)