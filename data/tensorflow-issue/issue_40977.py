from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import backend as K

class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, masking_boolean, vocab, dimension, **kwargs):
        self.masking_boolean = masking_boolean
        self.vocab = vocab
        self.dimension = dimension
        super(CustomEmbedding, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.vocab, self.dimension),
                                      initializer='glorot_uniform', dtype='float32',
                                      trainable=True)
        super(CustomEmbedding, self).build(input_shape)
    def call(self, inputs):
        return tf.nn.embedding_lookup(params=self.kernel, ids=inputs)
    def compute_mask(self, inputs, mask=None):
        return self.masking_boolean

    def compute_output_shape(self, input_shape):
        return (K.shape(input_shape)[0], K.shape(input_shape)[1], self.dimension)

def main():
    def create_model():
        sentence = tf.keras.layers.Input(dtype='int32', shape=(None,), name='sentence')
        pos_ids_mask_boolean = K.not_equal(0, sentence)
        pos_ids_mask_float = K.cast(pos_ids_mask_boolean, dtype='float32')
        a = CustomEmbedding(pos_ids_mask_boolean, 500, 512)(sentence)
        a = a*pos_ids_mask_float
        model = tf.keras.models.Model(inputs=[sentence], outputs=a)
        return model
    model = create_model()

if __name__ == '__main__':
    main()

import tensorflow as tf
from tensorflow.keras import backend as K

class CustomEmbedding(tf.keras.layers.Layer):
    def __init__(self, masking_boolean, vocab, dimension, **kwargs):
        self.masking_boolean = masking_boolean
        self.vocab = vocab
        self.dimension = dimension
        super(CustomEmbedding, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.vocab, self.dimension),
                                      initializer='glorot_uniform', dtype='float32',
                                      trainable=True)
        super(CustomEmbedding, self).build(input_shape)
    def call(self, inputs):
        return tf.nn.embedding_lookup(params=self.kernel, ids=inputs)
    def compute_mask(self, inputs, mask=None):
        return self.masking_boolean

    def compute_output_shape(self, input_shape):
        return (K.shape(input_shape)[0], K.shape(input_shape)[1], self.dimension)

def main():
    def create_model():
        sentence = tf.keras.layers.Input(dtype='int32', shape=(None,), name='sentence')
        pos_ids_mask_boolean = K.not_equal(0, sentence)
        pos_ids_mask_float = K.cast(K.not_equal(0, sentence), dtype='float32')
        a = CustomEmbedding(pos_ids_mask_boolean, 500, 512)(sentence)
        a = a*pos_ids_mask_float
        model = tf.keras.models.Model(inputs=[sentence], outputs=a)
        return model
    model = create_model()

if __name__ == '__main__':
    main()