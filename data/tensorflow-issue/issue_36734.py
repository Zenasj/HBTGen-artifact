from tensorflow.keras import layers
from tensorflow.keras import models

from __future__ import print_function
from __future__ import absolute_import

import logging
from mimic3newmodels.biobert.biobertlayer import BioBertLayer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


log = logging.getLogger('tensorflow')
log.handlers = []


class Network(Model):

    def __init__(self, dim, task, num_classes=1, input_dim=76, **kwargs):

        self.dim = dim
        
        if task in ['decomp', 'ihm', 'ph']:
            final_activation = 'sigmoid'
        elif task in ['los']:
            if num_classes == 1:
                final_activation = 'relu'
            else:
                final_activation = 'softmax'
        else:
            raise ValueError("Wrong value for task")

        print("==> not used params in network class:", kwargs.keys())
        
        X = Input(shape=(None, input_dim), name='X')
        inputs = [X]
        
        encoder = BioBertLayer(bert_path="mimic3newmodels/biobert/bert-module/", seq_len=48, tune_embeddings=False,
                    pooling='cls', n_tune_layers=3, verbose=False)
                    
        pred = tf.keras.layers.Dense(num_classes, activation=final_activation)(encoder(inputs))
        outputs = [pred]

        super(Network, self).__init__(inputs=inputs, outputs=outputs)
	
    def say_name(self):
        return "{}.n{}".format('biobert_pubmed',
                                self.dim,
                                )