from tensorflow import keras

import tensorflow as tf
import tensorflow_hub as hub

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = tf.keras.Sequential(
        [
            hub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2",
                output_shape=[2048],
                trainable=True,
            )
        ]
    )