import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = tf.keras.Input(shape=[100])
x = tf.keras.layers.Dense(10)(inputs)
x = tf.keras.layers.Dense(15)(x)
x = tf.keras.layers.Dense(5)(x)
outputs = tf.keras.layers.Dense(5)(x)

model = tf.keras.Model(inputs, outputs)

assert model.losses == []

l2_regularizer = tf.keras.regularizers.l2(1e-4)
for i in range(len(model.layers)):
    layer = model.layers[i]
    if isinstance(layer, tf.keras.layers.Dense):
        model.add_loss(lambda: l2_regularizer(layer.kernel))
        
print(model.losses)

inputs = tf.keras.Input(shape=[100])
x = tf.keras.layers.Dense(10)(inputs)
x = tf.keras.layers.Dense(15)(x)
x = tf.keras.layers.Dense(5)(x)
outputs = tf.keras.layers.Dense(5)(x)

model = tf.keras.Model(inputs, outputs)

assert model.losses == []

l2_regularizer = tf.keras.regularizers.l2(1e-4)
model.add_loss(lambda: l2_regularizer(model.layers[1].kernel))
model.add_loss(lambda: l2_regularizer(model.layers[2].kernel))
model.add_loss(lambda: l2_regularizer(model.layers[3].kernel))
model.add_loss(lambda: l2_regularizer(model.layers[4].kernel))
        
print(model.losses)

inputs = tf.keras.Input(shape=[100])
x = tf.keras.layers.Dense(10)(inputs)
x = tf.keras.layers.Dense(15)(x)
x = tf.keras.layers.Dense(5)(x)
outputs = tf.keras.layers.Dense(5)(x)

model = tf.keras.Model(inputs, outputs)

assert model.losses == []

l2_regularizer = tf.keras.regularizers.l2(1e-4)

def add_l2_regularization(layer):
    def _add_l2_regularization():
        l2 = tf.keras.regularizers.l2(1e-4)
        return l2(layer.kernel)
    return _add_l2_regularization

for i in range(len(model.layers)):
    layer = model.layers[i]
    if isinstance(layer, tf.keras.layers.Dense):
        model.add_loss(add_l2_regularization(layer))

model.losses