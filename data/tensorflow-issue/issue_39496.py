from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CustomModel(tf.keras.layers.Layer):
    #this class is for source sequence
    def __init__(self,):
        super(CustomModel, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(32, 512, 512),
                                      initializer=tf.keras.initializers.glorot_uniform(seed=1),
                                      trainable=True) #dumb
    def call(self, inputs):
        return inputs[0]

def main():
    def create_model(source_vocab, target_vocab, relationship_vocab):
        source = tf.keras.layers.Input(dtype='int32', shape=(64,), name='source')
        target = tf.keras.layers.Input(dtype='int32', shape=(64,), name='target')
        relationship = tf.keras.layers.Input(dtype='int32', shape=(1,), name='relationship')
        embedding_source = tf.keras.layers.Embedding(512, 512, input_length=64)(source)
        embedding_target = tf.keras.layers.Embedding(512, 512, input_length=64)(target)
        final_layer = CustomModel()([relationship, embedding_source, embedding_target])
        model = tf.keras.models.Model(inputs=[source, target, relationship], outputs=final_layer)
        return model
    model = create_model(1000, 1000, 500)
    print(model.summary())

if __name__ == '__main__':
    main()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class CustomModel(tf.keras.layers.Layer):
    #this class is for source sequence
    def __init__(self,):
        super(CustomModel, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(32, 512, 512),
                                      initializer=tf.keras.initializers.glorot_uniform(seed=1),
                                      trainable=True) #dumb
    def call(self, inputs):
        return inputs[0]

def main():
    def create_model(source_vocab, target_vocab, relationship_vocab):
        source = tf.keras.layers.Input(dtype='int32', shape=(64,), name='source')
        target = tf.keras.layers.Input(dtype='int32', shape=(64,), name='target')
        relationship = tf.keras.layers.Input(dtype='int32', shape=(1,), name='relationship')
        embedding_source = tf.keras.layers.Embedding(512, 512, input_length=64)(source)
        embedding_target = tf.keras.layers.Embedding(512, 512, input_length=64)(target)
        final_layer = CustomModel()([embedding_source, embedding_target, relationship])
        model = tf.keras.models.Model(inputs=[source, target, relationship], outputs=final_layer)
        return model
    model = create_model(1000, 1000, 500)
    print(model.summary())

if __name__ == '__main__':
    main()