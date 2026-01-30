from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

# Build a sequential and functional model
original_input = tf.keras.Input(shape=(1,), dtype=tf.float32)
sequential_model = tf.keras.Sequential([
    original_input,
    # tf.keras.layers.Lambda(lambda x: x),
])
functional_model = tf.keras.Model(sequential_model.inputs, sequential_model.outputs)

# Predict some data
print("sequential_model:", sequential_model.predict([[1.5]]))
print("functional_model:", functional_model.predict([[1.5]]))

# Clone the model
cloned_sequential_model = tf.keras.models.clone_model(sequential_model)
cloned_functional_model = tf.keras.models.clone_model(functional_model)
assert id(cloned_sequential_model.input) != id(original_input)  # OK
assert id(cloned_functional_model.input) != id(original_input)  # OK

# Clone the model and make the input expect a different shape and dtype
new_input = tf.keras.Input(shape=(None,), dtype=tf.uint8)
cloned_sequential_model = tf.keras.models.clone_model(sequential_model, input_tensors=[new_input])
cloned_functional_model = tf.keras.models.clone_model(functional_model, input_tensors=[new_input])

assert id(cloned_sequential_model.input) == id(new_input) # OK
assert id(cloned_functional_model.input) == id(new_input) # FAILS (expected to pass)
assert id(cloned_functional_model.input) != id(original_input)  # FAILS (expected to pass)

print("cloned_sequential_model.inputs:", cloned_sequential_model.inputs)
print("cloned_functional_model.inputs:", cloned_functional_model.inputs)

assert(cloned_sequential_model.predict([[1.5]]) == cloned_functional_model.predict([[1.5]])) # FAILS (expected to pass)
print("cloned_sequential_model:", cloned_sequential_model.predict([[1.5]]))
print("cloned_functional_model:", cloned_functional_model.predict([[1.5]]))

## Chain example
x = tf.keras.Input(shape=(1,), dtype=tf.float32)
y = tf.keras.layers.Lambda(lambda x: -x)(x)
other_model = tf.keras.Model(x, y)

chained_models = tf.keras.models.clone_model(functional_model, other_model.outputs)
assert chained_models.predict([[1]]) == [[-1]] # FAILS (expected to pass)
print("chained_models:", chained_models.predict([[1]]))