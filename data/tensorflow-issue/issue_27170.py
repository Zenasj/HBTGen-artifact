from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
if not hasattr(tf, 'nest'):
    tf.nest = tf.contrib.framework.nest


values = Input(shape=(10,), dtype=tf.float32)

## non-ragged version
indices = Input(shape=(5,), dtype=tf.int64)
gathered = Lambda(lambda args: tf.gather(*args))([values, indices])
Model(inputs=(values, indices), outputs=gathered)
# works fine

## ragged version
index_values = Input(shape=(), dtype=tf.int64)
index_row_splits = Input(shape=(), dtype=tf.int64)

indices = tf.RaggedTensor.from_row_splits(index_values, index_row_splits)
gathered = Lambda(lambda args: tf.gather(*args))([values, indices])
Model(inputs=(values, indices), outputs=gathered)
# raises
# 1.13.1: ValueError: Input tensors to a Model must come from `tf.keras.Input`.
# Received: tf.RaggedTensor(values=Tensor("input_3:0", shape=(?,), dtype=int64),
# row_splits=Tensor("input_4:0", shape=(?,), dtype=int64))
# (missing previous layer metadata).
# 2.0: AttributeError: 'RaggedTensor' object has no attribute 'op'

def ragged_tensor_from_row_lengths(values, row_lengths):
    def components(args):
        values, row_lengths = args
        ragged = tf.RaggedTensor.from_row_lengths(values, row_lengths)
        return ragged.values, ragged.row_splits

    components = tf.keras.layers.Lambda(components)([values, row_lengths])
    return tf.RaggedTensor.from_row_splits(*components)


def as_ragged_components(tensor):
    if isinstance(tensor, tf.RaggedTensor):
        return dict(values=tensor.values, row_splits=tensor.row_splits)
    elif isinstance(tensor, (list, tuple)):
        return tuple(as_ragged_components(t) for t in tensor)
    elif isinstance(tensor, dict):
        return {k: as_ragged_components(v) for k, v in tensor.items()}
    else:
        # leave unchanged
        assert(isinstance(tensor, tf.Tensor))
        return tensor


def as_ragged(components):
    if isinstance(components, (list, tuple)):
        return tuple(as_ragged(c) for c in components)
    elif isinstance(components, dict):
        if all(k in components for k in ('values', 'row_splits')):
            return tf.RaggedTensor.from_row_splits(**components)
        else:
            return {k: as_ragged(v) for k, v in components.items()}
    else:
        assert(isinstance(components, tf.Tensor))
        return components


def ragged_lambda(fn, args):
    assert(isinstance(args, (list, tuple)))
    if not any(isinstance(a, tf.RaggedTensor) for a in args):
        out_components = tf.keras.layers.Lambda(fn)(args)
    else:
        components = as_ragged_components(args)
        flat_args = tf.nest.flatten(components)

        def actual_fn(flat_args):
            args = tf.nest.pack_sequence_as(components, flat_args)
            args = as_ragged(components)
            out = fn(args)
            return as_ragged_components(out)

        out_components = tf.keras.layers.Lambda(actual_fn)(flat_args)
    return as_ragged(out_components)


gathered = ragged_lambda(
    lambda args: tf.gather(*args), [values, indices])

gathered_components = tf.nest.flatten(as_ragged_components(gathered))
Model(
    inputs=(values, index_values, index_row_splits),
    outputs=gathered_components)

import tensorflow as tf

class RaggedModelOutputTest(tf.test.TestCase):

    def test_simple(self):
        inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
        output = tf.expand_dims(inp, axis=-1)
        tf.keras.Model(inputs=inp, outputs=output)
        # looks like we don't have to wrap everything in lambdas anymore!

    def test_ragged_conversion_at_end(self):
        inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
        output = tf.RaggedTensor.from_tensor(inp)
        tf.keras.Model(inputs=inp, outputs=output)  # <--- fails
        # oh ffs

    def test_ragged_wrapped_lambda(self):
        inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
        output = tf.keras.layers.Lambda(tf.RaggedTensor.from_tensor)(inp)
        tf.keras.Model(inputs=inp, outputs=output)
        # not the hardest work-around, but took me a while to work out...


tf.test.main()

import tensorflow as tf

flat = tf.keras.layers.Input(shape=(2,), ragged=True)
row_splits = tf.keras.layers.Input(shape=(), dtype=tf.int64)
rt = tf.RaggedTensor.from_row_splits(flat, row_splits)

# fails
# ------------------------------------------------------------
# x = tf.reduce_max(rt, axis=1)
# ------------------------------------------------------------

# fails
# ------------------------------------------------------------
# x = tf.keras.layers.Lambda(tf.reduce_max, arguments=dict(axis=1))(rt)
# ------------------------------------------------------------

# Succeeds
# ------------------------------------------------------------
def ragged_max(args, axis):
    flat_values, row_splits = args
    return tf.reduce_max(
        tf.RaggedTensor.from_row_splits(flat_values, row_splits), axis=axis)


x = tf.keras.layers.Lambda(ragged_max, arguments=dict(axis=1))(
    [flat, row_splits])
# ------------------------------------------------------------

model = tf.keras.models.Model(inputs=(flat, row_splits), outputs=x)

model.save_weights('/tmp/ragged-model.h5')
print('Saved successfully')

import tensorflow as tf

val_ragged = tf.ragged.constant([[1, 2, 3], [1, 2], [1, 2, 3, 4]])

val_tensor = val_ragged.to_tensor()

inputs = tf.keras.layers.Input(shape=(None, None,), ragged=False)
outputs = tf.keras.layers.Embedding(5, 4)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# this model with normal tensor works
print(model(val_tensor))

inputs_ragged = tf.keras.layers.Input(shape=(None, None,), ragged=True)
outputs_ragged = tf.keras.layers.Embedding(5, 4)(inputs_ragged)
model_ragged = tf.keras.Model(inputs=inputs_ragged, outputs=outputs_ragged)

# this one with RaggedTensor doesn't
print(model_ragged(val_ragged))

import tensorflow as tf

val_ragged = tf.ragged.constant([[1, 2, 3], [1, 2], [1, 2, 3, 4]])
y = tf.constant([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
val_tensor = val_ragged.to_tensor()

# Tensor version
inputs = tf.keras.layers.Input(shape=(None, ), ragged=False)
embeddings = tf.keras.layers.Embedding(5, 4)(inputs)
lstm_out = tf.keras.layers.LSTM(8)(embeddings)
classifier = tf.keras.layers.Dense(
        2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)
model.compile(
    tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy())
_ = model(val_tensor)
model.fit(val_tensor, y, epochs=1)

# Ragged Tensor version
inputs = tf.keras.layers.Input(shape=(None, ), ragged=True)
embeddings = tf.keras.layers.Embedding(5, 4)(inputs)
lstm_out = tf.keras.layers.LSTM(8)(embeddings)
classifier = tf.keras.layers.Dense(
        2, activation='softmax', name='classifier')(lstm_out)
model = tf.keras.Model(inputs=inputs, outputs=classifier)
model.compile(
    tf.keras.optimizers.Adam(0.01),
    loss=tf.keras.losses.CategoricalCrossentropy())
_ = model(val_ragged)  # WORKS UNTIL THIS POINT INCLUDED. We can calculate the model output
model.fit(val_ragged, y, epochs=1)  # Fails here

pooled_embeddings = tf.keras.layers.Lambda(tf.reduce_mean, arguments=dict(axis=1))(embeddings)