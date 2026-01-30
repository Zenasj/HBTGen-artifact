from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf


INPUT_BX_NAME = "bx"


# NOTE: Encoder A
input_a0 = tf.keras.Input(shape=(1,), name="a0", dtype="float32")
input_a1 = tf.keras.Input(shape=(1,), name="a1", dtype="float32")
all_feature_a = tf.keras.layers.concatenate([input_a0, input_a1])
output_a = tf.keras.layers.Dense(4, activation="relu")(all_feature_a)
encoder_a = tf.keras.Model(inputs=[input_a0, input_a1], outputs=output_a)

# NOTE: Encoder B
input_b1 = tf.keras.Input(shape=(1,), name="b1", dtype="float32")
input_bx = tf.keras.Input(shape=(1,), name=INPUT_BX_NAME, dtype="string")
encoded_bx = tf.keras.layers.StringLookup(
    vocabulary=("a", "b", "c"), output_mode="one_hot"
)(input_bx)
all_feature_b = tf.keras.layers.concatenate([encoded_bx, input_b1])
output_b = tf.keras.layers.Dense(4, activation="relu")(all_feature_b)
encoder_b = tf.keras.Model(inputs=[input_bx, input_b1], outputs=output_b)

# NODE: Final model
a_inputs = encoder_a.inputs
b_inputs = encoder_b.inputs
encoded_a = encoder_a(a_inputs)
encoded_b = encoder_b(b_inputs)
prob = tf.keras.layers.Dense(1, activation="sigmoid")(tf.abs(encoded_a - encoded_b))
model = tf.keras.Model(inputs=[a_inputs, b_inputs], outputs=prob)

x_a = dict(
    a0=tf.constant([0, 1, 0, 1, 0, 1]),
    a1=tf.constant([1, 1, 0, 0, 0, 1]),
)
x_b = {
    "b1": tf.constant([1, 0, 0, 1, 0, 0]),
    INPUT_BX_NAME: tf.constant(["a", "b", "c", "a", "b", "a"]),
}
y = tf.constant([1, 0, 1, 0, 1, 0])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=["acc"],
)
model.fit([x_a, x_b], y)