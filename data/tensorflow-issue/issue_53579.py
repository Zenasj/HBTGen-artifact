import numpy as np
import tensorflow as tf

def ElmoEmbedding(x):
    return elmo_model.signatures["tokens"](tokens = tf.squeeze(tf.cast(x, tf.string)), sequence_len = tf.constant(batch_size * [max_len]))["elmo"]

def build_model(max_len, n_words, n_tags): 
    word_input_layer = Input(shape=(max_len, 40, ))
    elmo_input_layer = Input(shape=(max_len,), dtype=tf.string)
    word_output_layer = Dense(n_tags, activation = 'softmax')(word_input_layer)
    elmo_output_layer = Lambda(ElmoEmbedding, output_shape=(None, 1024))(elmo_input_layer)

    output_layer = Concatenate()([word_output_layer, elmo_output_layer])
    output_layer = BatchNormalization()(output_layer)
    output_layer = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(output_layer)
    output_layer = TimeDistributed(Dense(n_tags, activation='softmax'))(output_layer)
    
    model = Model([elmo_input_layer, word_input_layer], output_layer)
    
    return model

elmo_model = hub.load('https://tfhub.dev/google/elmo/3')
model = build_model(max_len, n_words, n_tags)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit([np.array(X1_train), np.array(X2_train).reshape((len(X2_train), max_len, 40))], 
                    y_train, 
                    validation_data=([np.array(X1_valid), np.array(X2_valid).reshape((len(X2_valid), max_len, 40))], y_valid),
                    batch_size=32, epochs=2, verbose=1)