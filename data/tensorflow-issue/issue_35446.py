import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

def test_model_with_scalar_input():
    model_home = "models"
    input_shape = 5, 10, 128
    n_class = 5

    batch_size, seq_len, fea_size = input_shape

    input_fea = tf.keras.layers.Input(shape=input_shape[1:])
    input_seq = tf.keras.layers.Input(shape=())

    name = "test-model"
    forward_layer = tf.keras.layers.LSTM(units=fea_size, return_sequences=True, name=name + "-forward")

    backward_layer = tf.keras.layers.LSTM(units=fea_size, return_sequences=True, go_backwards=True,
                                          name=name + "-backward")

    bi_lstm = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer, name=name + "-bi-lstm")

    fc = tf.keras.layers.Dense(units=n_class, activation="softmax", name="fc-pred")

    mask = tf.sequence_mask(input_seq, seq_len)

    y = bi_lstm(input_fea, mask=mask)
    y = fc(y)

    model = tf.keras.models.Model(inputs=[input_fea, input_seq], outputs=y, name=name)

    features = np.random.normal(size=[batch_size * 100, seq_len, fea_size])
    labels = np.random.randint(0, n_class, size=[batch_size * 100, seq_len])
    # seq = np.random.randint(0, seq_len, size=[batch_size * 100])
    seq = [seq_len] * (batch_size * 100)

    dataset = tf.data.Dataset.from_tensor_slices(((features, seq), labels))
    dataset = dataset.batch(batch_size)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    saver = tf.keras.callbacks.ModelCheckpoint(model_home, monitor="loss", verbose=1, save_best_only=True)
    model.fit(dataset, epochs=5, callbacks=[saver], steps_per_epoch=10)

    new_model = tf.keras.models.load_model(model_home)

    # print(new_model.layers)

    pred = new_model.predict(dataset)
    pred = np.argmax(pred, axis=-1)
    print(pred)