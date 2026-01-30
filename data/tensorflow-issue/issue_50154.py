import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.python.keras.layers import Masking, Bidirectional, LSTM, Dropout, TimeDistributed, Dense, Lambda
from tensorflow_addons.metrics import F1Score
import numpy as np

window_length = 50
embedding_dimension = 100
num_classes = 5

tf.random.set_seed(42)
embeddings = tf.keras.Input(shape=(window_length, embedding_dimension), dtype=tf.float32, name="embedding_sequence")
nwords = tf.keras.Input(shape=(), dtype=tf.int32, name="nwords")

masked_embedding = Masking()(embeddings)

bilstm = Bidirectional(
    LSTM(
        units=16,
        return_sequences=True,
        dropout=0.0,
        recurrent_dropout=0.0,
    )
)(masked_embedding)

bilstm = Dropout(rate=0.5, seed=42)(bilstm)

logits = TimeDistributed(Dense(num_classes, activation="softmax"), name="logits")(bilstm)

pred_ids = tf.argmax(logits, axis=2, output_type=tf.int32)

naming_layer = Lambda(lambda x: x, name="pred_ids")
pred_ids = naming_layer(pred_ids)

loss = {"logits": "categorical_crossentropy"}

model = tf.keras.Model(inputs=[embeddings, nwords], outputs=[logits, pred_ids], name="ner_bilstm")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate= 0.001),
    loss=loss,
    metrics={"logits": [F1Score(num_classes=num_classes, average="micro")]},
)

dummy_pred_ids = np.array(["dummy"] * 7)



data_x = {"embedding_sequence": np.random.rand(7,window_length,embedding_dimension), "nwords": np.array([window_length]*7)}
data_y = {
    "logits": np.zeros((7, window_length, num_classes), dtype=int),
    "pred_ids": dummy_pred_ids,
}
model.fit(x=data_x, y=data_y, validation_data=(data_x, data_y))