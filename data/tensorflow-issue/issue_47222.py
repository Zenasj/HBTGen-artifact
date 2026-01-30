import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

x1 = tf.ragged.constant([[1, 2, 3, 4, 5], [8, 9], [10, 11, 12, 13], [1, 2, 3]])
y1 = tf.constant([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0],])
model1 = tf.keras.Sequential([
                          tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
                          tf.keras.layers.Embedding(15, 4),
                          tf.keras.layers.LSTM(units=64, activation='tanh', dropout=0.2, name = 'LSTM_1', return_sequences=True),
                          tf.keras.layers.LSTM(units=256, activation='tanh', dropout=0.2, name = 'LSTM_2',return_sequences=True),
                          tf.keras.layers.LSTM(units=100, activation='tanh', dropout=0.2, name = 'LSTM_3', return_sequences=False),
                          tf.keras.layers.Dense(units=4, activation='sigmoid', name = 'Dense_1')
])


model1.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model1.summary()
model1.fit(x1, y1, batch_size=2)

model = keras.Sequential([
                          tf.keras.layers.Input(shape=[None], dtype=tf.float32, ragged=True),
                          tf.keras.layers.Embedding(18286, 4),
                          keras.layers.LSTM(units=64, activation='tanh', dropout=0.2, name = 'LSTM_1', return_sequences=True),
                          keras.layers.LSTM(units=256, activation='tanh', dropout=0.2, name = 'LSTM_2',return_sequences=True),
                          keras.layers.LSTM(units=100, activation='tanh', dropout=0.2, name = 'LSTM_3', return_sequences=False),
                          # tf.keras.layers.LSTM(4),
                          #tf.keras.layers.Flatten(name = 'Flatten'),
                          tf.keras.layers.Dense(units=4, activation='sigmoid', name = 'Dense_1')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

model.fit(tf_X_train, y, epochs=20, batch_size=2)