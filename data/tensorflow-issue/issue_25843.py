import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(vocab_size=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 80),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

model = create_model(vocab_size=vocab_size)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


dataset1 = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5,6]])
dataset2 = tf.data.Dataset.from_tensor_slices([1, 2])
dataset = tf.data.Dataset.zip((dataset1, dataset2))
model.fit(x=dataset,
          epochs=1,
          verbose=1)