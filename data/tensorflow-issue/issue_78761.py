import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import os

tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def Model_WdZYze5JTnF3OjiUHVAKw_iFV2jkQLyL(input):
    input = tf.keras.Input(shape=input)
    _input = input
    _zeropadding_input = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (0, 0)))(input)
    conv1_output = tf.keras.layers.Conv2DTranspose(filters=6, kernel_size=(5, 5), strides=(1, 1), padding="valid", data_format="channels_last", dilation_rate=(1, 1), use_bias=True, dtype=tf.float16, name="conv1_mutated")(input)
    output_transpose = [(0), (0, 1), (0, 2, 1), (0, 3, 1, 2), (0, 4, 1, 2, 3)]
    tail_flatten_output = tf.keras.layers.Flatten(name="tail_flatten", dtype=tf.float32)(conv1_output)
    tail_fc_output = tf.keras.layers.Dense(units=10, use_bias=True, dtype=tf.float64, name="tail_fc")(tail_flatten_output)

    tail_fc_output = tail_fc_output
    model = tf.keras.models.Model(inputs=_input, outputs=tail_fc_output)
    return model

inp = np.random.random([1, 1, 28, 28])
tf_input = tf.convert_to_tensor(inp.transpose(0, 2, 3, 1))
tf_model = Model_WdZYze5JTnF3OjiUHVAKw_iFV2jkQLyL(tf_input.shape[1:])
tf_output = tf_model(tf_input)
print(tf_output)