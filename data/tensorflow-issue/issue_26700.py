import random

import unittest
import numpy as np
from tensorflow.python import keras as tfk
import tensorflow as tf

#tf.enable_v2_behavior()
tf.enable_eager_execution()

class EagerWeightsTest(unittest.TestCase):

    def test_eager_weights(self):
        weights = np.zeros((10, 11))                           # use zero weights
        dense = tfk.layers.Dense(units=11, input_shape=(10,),
                                 weights=[weights],
                                 use_bias=False)

        model = tfk.models.Sequential()
        model.add(dense)
        model.compile("adam", tfk.losses.mae)

        logits = model.predict(np.random.random((3, 10)))

        self.assertTrue(np.allclose(np.zeros((3, 11)),          # expect zero outputs
                                    logits))