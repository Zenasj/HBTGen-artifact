import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class MyWordEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(300, 512), dtype='float32')
        super(MyWordEmbedding, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        return tf.nn.embedding_lookup(params=self.kernel, ids=inputs[0])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, mask_para, **kwargs):
        self.mask_para = mask_para
        super(EncoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Qdense = self.add_weight(name='Qdense', shape=(512, 512))
        super(EncoderLayer, self).build(input_shape)

    def call(self, x):
        Qoutput = tf.einsum('aij,jk->aik', x[0], self.Qdense)
        Koutput =  tf.einsum('aij,jk->aik', x[0], self.Qdense)
        Voutput =  tf.einsum('aij,jk->aik', x[0], self.Qdense)
        a = tf.einsum('ajk,afk->ajf', Qoutput, Koutput) * tf.tile(K.expand_dims(self.mask_para, axis=1), [1, 64, 1])
        a = tf.matmul(a, Voutput)
        return a

    def compute_mask(self, inputs, mask):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def create_encoder_model():
    word_ids_fr = tf.keras.layers.Input(dtype='int32', shape=(None,))
    a = MyWordEmbedding()([word_ids_fr])
    a = EncoderLayer(K.cast(K.not_equal(0, word_ids_fr), dtype='float32'))([a])
    model = tf.keras.models.Model(inputs=[word_ids_fr], outputs=a)
    return model

def create_model():
    word_ids_en = tf.keras.layers.Input(dtype='int32', shape=(None,))
    a = tf.keras.Input(shape=(None, 512,))
    b = MyWordEmbedding()([word_ids_en])
    b = b + a
    model = tf.keras.models.Model(inputs=[word_ids_en, a], outputs=b)
    return model
    
def evaluate():
    source_sequence_ids = pad_sequences(np.random.randint(5, size=(3, 64)), maxlen=64, padding='pre')
    output = decoder_model.predict([pad_sequences(np.random.randint(5, size=(3, 64)), maxlen=64, padding='post'), encoder_model(source_sequence_ids, training=False)], steps=1, verbose=1, batch_size=256)

decoder_model = create_model()
encoder_model = create_encoder_model()
evaluate()

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class MyWordEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(300, 512), dtype='float32')
        super(MyWordEmbedding, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, inputs):
        return tf.nn.embedding_lookup(params=self.kernel, ids=inputs[0])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Qdense = self.add_weight(name='Qdense', shape=(512, 512))
        super(EncoderLayer, self).build(input_shape)

    def call(self, x):
        Qoutput = tf.einsum('aij,jk->aik', x[0], self.Qdense)
        Koutput =  tf.einsum('aij,jk->aik', x[0], self.Qdense)
        Voutput =  tf.einsum('aij,jk->aik', x[0], self.Qdense)
        mask_para = x[1]
        a = tf.einsum('ajk,afk->ajf', Qoutput, Koutput) * tf.tile(K.expand_dims(mask_para, axis=1), [1, 64, 1])
        a = tf.matmul(a, Voutput)
        return a

    def compute_mask(self, inputs, mask):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def create_encoder_model():
    word_ids_fr = tf.keras.layers.Input(dtype='int32', shape=(None,))
    a = MyWordEmbedding()([word_ids_fr])
    a = EncoderLayer()([a, K.cast(K.not_equal(0, word_ids_fr), dtype='float32')])
    model = tf.keras.models.Model(inputs=[word_ids_fr], outputs=a)
    return model

def create_model():
    word_ids_en = tf.keras.layers.Input(dtype='int32', shape=(None,))
    a = tf.keras.Input(shape=(None, 512,))
    b = MyWordEmbedding()([word_ids_en])
    b = b + a
    model = tf.keras.models.Model(inputs=[word_ids_en, a], outputs=b)
    return model
    
def evaluate():
    source_sequence_ids = pad_sequences(np.random.randint(5, size=(3, 64)), maxlen=64, padding='pre')
    output = decoder_model.predict([pad_sequences(np.random.randint(5, size=(3, 64)), maxlen=64, padding='post'), encoder_model(source_sequence_ids, training=False)], steps=1, verbose=1, batch_size=256)

decoder_model = create_model()
encoder_model = create_encoder_model()
evaluate()