from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import keras
import keras.layers

class CosineSimilarityLayer(keras.layers.Layer):
    def __init__(
        self, num_classes: int, name: str = None
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self._weights = None

    def build(self, input_shape):
        self._weights = self.add_weight(
            name="W",
            shape=(
                input_shape[-1],
                self.num_classes,
            ),
            initializer="glorot_normal",
            trainable=True,
            dtype=self.dtype
        )
        super().build(input_shape)

    def compute_output_shape(self):
        return None, self.num_classes

    # If you comment out tf.function here, it works.
    @tf.function
    def call(self, inputs: tf.Tensor):
        embedding = inputs
        # normalize feature
        embedding_normalized = tf.nn.l2_normalize(embedding, axis=1)
        # get centroids
        weights_normalized = tf.nn.l2_normalize(self._weights, axis=1, )

        logits = embedding_normalized @ weights_normalized

        return logits

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_classes": self.num_classes,
            }
        )
        return config

def main():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    layer = CosineSimilarityLayer(num_classes=100)

    input = tf.zeros(shape=(10, 100), dtype=tf.float16)

    model = keras.models.Sequential([
        layer
    ])

    model(input)

    model.save("autocast_issue")

if __name__ == "__main__":
    main()