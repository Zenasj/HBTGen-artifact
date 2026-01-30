from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.python import debug as tf_debug
keras.backend.set_session(\
        tf_debug.LocalCLIDebugWrapperSession(tf.compat.v1.Session()))

import sys
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def basic_generator():
    """
    """
    data_x = [[0, 1, 2, 3], [1, 2, 3, 4], \
        [0, 1, 3, 4], [0, 2, 1, 4]]
    data_y = [[0, 1, 2, 3], [1, 2, 3, 4], \
        [0, 1, 3, 4], [0, 2, 1, 4]]
    for x, y in zip(data_x, data_y):
        yield (x, y)

if __name__ == "__main__":
    params = { 
       'seq_len': 8,
        'use_bert': False,
        'batch_size': 2,
        'epochs': 5,
        'vocab_len': 5,
        'num_entities': 3,
        'use_pre_emb': False,
        'emb_dim': 20,
        'use_emb_drop': False,
        'emb_drop_rate': 0.8,
        'num_layer': 1,
        'num_lstm_cell': [15],
        'rec_drops': [0.8],
        'lr': 0.001
    }

    output_types = (tf.int32, tf.int32)
    dataset = tf.data.Dataset.from_generator(\
        basic_generator, output_types)
    dataset = dataset.padded_batch(\
        params['batch_size'], \
            ([params['seq_len'], ], [params['seq_len'], ]))

    from tensorflow.python import debug as tf_debug
    keras.backend.set_session(\
        tf_debug.LocalCLIDebugWrapperSession(tf.compat.v1.Session()))

    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, params['num_entities'])))
    dataset = dataset.repeat(params['epochs'])
    
    #model = LstmCrf(params)
    model = keras.Sequential()
    inputs = keras.Input(shape=(None, ))
    emb_ini = keras.initializers.TruncatedNormal()
    model.add(keras.layers.Embedding(\
                params['vocab_len'], params['emb_dim'], \
                    embeddings_initializer=emb_ini, mask_zero=True))        
    model.add(keras.layers.TimeDistributed(\
        keras.layers.Dense(params['num_entities'])))
    model.add(keras.layers.Activation('softmax'))
       
    model.compile(optimizer=keras.optimizers.Adam(params['lr']), \
        loss='categorical_crossentropy', \
            metrics=['accuracy'])

    model.fit(dataset, epochs=params['epochs'], \
        steps_per_epoch=2)