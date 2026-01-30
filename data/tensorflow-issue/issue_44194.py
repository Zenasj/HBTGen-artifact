import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

# Uncomment the following line to get proper GPU placement for the Embedding layer
# tf.compat.v1.disable_eager_execution()
tf.debugging.set_log_device_placement(True)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(10, 1, input_shape=(1,)),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# Train on some dummy data
x = np.random.randint(1, 10, (10, 1))
y = np.random.uniform(0, 1, (10, 1))
model.fit(x, y, epochs=12)

class GPUCompatibleEmbedding(layers.Embedding):
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )
        self.built = True