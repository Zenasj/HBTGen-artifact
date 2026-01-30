import numpy as np
from tensorflow.keras import optimizers

py
train_text = [['a'],['a'],['a']]
val_text = [['a'],['a'],['a']]
train_label = np.asarray([5,5,5])
val_label = np.asarray([5,5,5])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Implementing a Transformer block as a layer

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

vocab_size = 40000  # Only consider the top 20k words :)
maxlen = 10  # for my task, ideally 500

train_label = tf.keras.utils.to_categorical(train_label, 20)
val_label = tf.keras.utils.to_categorical(val_label, 20)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, lower=True, oov_token='<OOV>')
tokenizer.fit_on_texts(train_text)
train_sequences = tokenizer.texts_to_sequences(train_text)

val_sequences = tokenizer.texts_to_sequences(val_text)

train_text = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=maxlen, dtype='int32',
                                           padding='post', truncating='post')

val_text = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=maxlen, dtype='int32',
                                           padding='post', truncating='post')

from sklearn.preprocessing import normalize

train_text_n = normalize(train_text)
val_text_n = normalize(val_text)

data_set = tf.data.Dataset.from_tensor_slices((train_text_n , train_label))
data_set

from tensorflow.keras.regularizers import l2
 
embed_dim = 128  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 256  # Hidden layer size in feed forward network inside transformer
batch_size = 32
factor = 0.0001
 
inputs = layers.Input(shape=(maxlen,))

embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

x = transformer_block(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(160, activation="relu", kernel_initializer='glorot_uniform')(x)
x = layers.Dropout(0.15)(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(90, activation="relu", kernel_initializer='glorot_uniform')(x)
x = layers.Dropout(0.1)(x)
x = layers.BatchNormalization()(x)

 
x = layers.Dense(40, activation="relu", kernel_initializer='glorot_uniform')(x)
x = layers.Dropout(0.1)(x)

outputs = layers.Dense(20)(x)
 
model = keras.Model(inputs=inputs, outputs=outputs)
 
adamopt = tf.keras.optimizers.Adam(learning_rate=1e-4)
 
 
model.compile(optimizer='adamopt', loss="categorical_crossentropy", metrics=["acc"])
 
model.summary()
 
history = model.fit(
    train_text_n, train_label, batch_size=batch_size, epochs=20, validation_data=(val_text_n, val_label), verbose=1)

py
outputs = layers.Dense(20,activation='sigmoid')(x)

output = np.zeros((20),dtype=np.float32)
output = np.put(sample, sample[0]+3 , 1 )