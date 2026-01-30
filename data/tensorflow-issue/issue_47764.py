import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

text_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
tokenize = hub.KerasLayer(preprocessor.tokenize)
tokenized_inputs = [tokenize(text_inputs)]
seq_length = 512
bert_pack_inputs = hub.KerasLayer(
    preprocessor.bert_pack_inputs,
    arguments=dict(seq_length=seq_length))
encoder_inputs = bert_pack_inputs(tokenized_inputs)
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1", trainable=True)
encoder_outputs = encoder(encoder_inputs)['pooled_output']
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(encoder_outputs)

model = tf.keras.Model(inputs=[text_inputs], outputs=[output_layer])
model.summary()


# class weights
target_labels = y_train.tolist()
class_weights = compute_class_weight(
    "balanced", classes=np.unique(target_labels), y=target_labels
)
class_weights = dict(zip(np.unique(target_labels), class_weights))

model.compile(tf.keras.optimizers.Adam(3e-05, epsilon=1e-08, clipnorm=1.0), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["acc"])

model.fit(x_train, y_train, batch_size=64, validation_data=(x_valid, y_valid), epochs=10, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)], class_weight=class_weights)