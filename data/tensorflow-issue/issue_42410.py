from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(tf.__version__)


class Embed(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.embed_dim],
            initializer=tf.keras.initializers.GlorotNormal(),
        )

    def call(self, inputs):
        return tf.gather(self.embeddings, tf.cast(inputs, tf.int32))


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed = Embed(vocab_size, embed_dim)
        self.dense = tf.keras.layers.Dense(350)

    def _build(self):
        self(tf.keras.Input([None], dtype=tf.int32))

    def call(self, inputs, training=False):
        outputs = self.embed(inputs, training=training)
        return self.dense(outputs, training=training)


model = Model(29, 320)
model._build()
model.summary()


@tf.function(
    input_signature=[
        tf.TensorSpec([1, None], dtype=tf.int32)
    ]
)
def func(inputs):
    i = tf.constant(0, dtype=tf.int32)
    T = tf.constant(100, dtype=tf.int32)

    def _cond(i, T): return tf.less(i, T)

    def _body(i, T):
        _ = model(inputs)
        return i + 1, T

    _, _ = tf.while_loop(
        _cond,
        _body,
        loop_vars=(i, T),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape([])
        )
    )

    return inputs


print(func(tf.zeros([1, 100], tf.int32)))

concrete_func = func.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite = converter.convert()