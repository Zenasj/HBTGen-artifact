import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import logging
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_decoder():
    """Build the decoder."""
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Dense(2))
    return decoder

def get_model(encoder):
    encoder.trainable = False
    decoder = get_decoder()
    inputs = encoder.input
    emb = encoder(inputs, training=False)
    outputs = decoder(emb)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model")

# local path to SavedModel
model_dir = "toy_model"
feats_dim = 64
distribute_strategy = tf.distribute.MirroredStrategy()
# build variables inside scope so they are mirrored across gpu replicas
with distribute_strategy.scope():
    # load pretrained encoder from tf SavedModel
    encoder = tf.keras.models.load_model(
        model_dir,
    ).get_layer('model')
    encoder.build(tf.TensorShape([None, None, feats_dim]))
    # Build decoder on top of frozen encoder
    model = get_model(encoder=encoder)
    model(tf.random.normal([1, 100, 64]))
    model.summary()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=opt
    )

encoder.build(tf.TensorShape([None, None, feats_dim]))
model(tf.random.normal([1, 100, 64]))