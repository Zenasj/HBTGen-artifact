import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

layer = tf.keras.layers.Dense(2, name="dense")
model = tf.keras.Sequential([layer], name="model")
print(model.built)
with tf.name_scope("parent"):
    with tf.name_scope(model.name):
        model.build((None, 3))
print([v.name for v in model.variables])

layer = tf.keras.layers.Dense(2, name="dense")
model = tf.keras.Sequential([layer], name="model")
print(model.built)
with tf.name_scope("parent"):
    model(tf.zeros((32, 3)))
print([v.name for v in model.variables])

layer = tf.keras.layers.Dense(2, name="dense")
model = tf.keras.Sequential([layer], name="model")
print(model.built)
with tf.name_scope("parent"):
    model(tf.keras.Input((2,)))
print([v.name for v in model.variables])

layer_built = tf.keras.layers.Dense(2, name="built")
layer_tensor_called = tf.keras.layers.Dense(2, name="tensor_called")
layer_input_called = tf.keras.layers.Dense(2, name="input_called")

inputs = tf.keras.Input((3,))

with tf.name_scope("parent"):
    
    with tf.name_scope(layer_built.name):
        layer_built.build((None, 3))
    
    layer_tensor_called(tf.zeros((32, 3)))
    
    layer_input_called(inputs)

print([v.name for v in layer_built.variables])
print([v.name for v in layer_tensor_called.variables])
print([v.name for v in layer_input_called.variables])

def Conv2D_BN(
    filters,
    kernel_size,
    strides=1,
    padding="same",
    activation="relu",
    use_bias=False,
    name=None,
    **kwargs,
):
    """Utility class to apply conv + BN.

    # Arguments
        x: input tensor.
        filters:
        kernel_size:
        strides:
        padding:
        activation:
        use_bias:

    Attributes
    ----------
    activation
        activation in `Conv2D`.
    filters
        filters in `Conv2D`.
    kernel_size
        kernel size as in `Conv2D`.
    padding
        padding mode in `Conv2D`.
    strides
        strides in `Conv2D`.
    use_bias
        whether to use a bias in `Conv2D`.
    name
        name of the ops; will become `name + '/Act'` for the activation
        and `name + '/BatchNorm'` for the batch norm layer.
    """
    if name is None:
        raise ValueError("name cannot be None!")

    layers = [
        Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=f"{name}/Conv2D",
        )
    ]

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        layers += [BatchNormalization(axis=bn_axis, scale=False, name=f"{name}/BatchNorm")]

    if activation is not None:
        layers += [Activation(activation, name=f"{name}/Act")]

    return tf.keras.Sequential(layers, name=name, **kwargs)

layer_built = tf.keras.layers.Dense(2, name="built")
layer_tensor_called = tf.keras.layers.Dense(2, name="tensor_called")
layer_input_called = tf.keras.layers.Dense(2, name="input_called")

inputs = tf.keras.Input((3,))

with tf.name_scope("parent"):

    with tf.name_scope(layer_built.name):
        layer_built.build((None, 3))

    layer_tensor_called(tf.zeros((32, 3)))

    layer_input_called(inputs)

print([v.name for v in layer_built.variables])
print([v.name for v in layer_tensor_called.variables])
print([v.name for v in layer_input_called.variables])