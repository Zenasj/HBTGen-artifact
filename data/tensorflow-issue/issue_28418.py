import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=10))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(50))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))


model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

es = tf.keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=10)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'],callback=[es])

model.fit(train_padded_docs, train_labels,validation_data=(val_padded_docs,val_labels), epochs=15, verbose=1,batch_size=10000)