import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(layers.Conv1D(filters=128, kernel_size=2, activation=activation, input_shape=(window_size // subseq_size, subseq_size)))
model.add(layers.Conv1D(filters=64, kernel_size=2, activation=activation))
model.add(layers.Conv1D(filters=32, kernel_size=2, activation=activation))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.TimeDistributed(Flatten()))
model.add(layers.LSTM(500))
model.add(layers.Dense(100))
model.add(layers.Dense(1))

opt = tf.train.AdamOptimizer(learning_rate)

tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, 
        strategy=tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(tpu = [TPU_ADDRESS1])))

tpu_model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mape', 'acc'])

H = tpu_model.fit(X, y, validation_split=0.15, epochs=epochs_n, batch_size = window_size // subseq_size)

mape

metrics

compile