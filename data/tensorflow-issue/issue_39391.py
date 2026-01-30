import numpy as np
from tensorflow.keras import layers

# Example 2: Sequential model
# Recreate the pretrained model, and load the saved weights.
inputs = keras.Input(shape=(784,), name='digits')
x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
pretrained_model = keras.Model(inputs=inputs, outputs=x, name='pretrained')

# Sequential example:
model = keras.Sequential(
    [pretrained_model, keras.layers.Dense(5, name='predictions')])
model.summary()

pretrained_model.load_weights('pretrained_ckpt')

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error,
# but will *not* work as expected. If you inspect the weights, you'll see that
# none of the weights will have loaded. `pretrained_model.load_weights()` is the
# correct method to call.

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error

# Warning! Calling `model.load_weights('pretrained_ckpt')` won't throw an error

for a, b in zip(pretrained.weights, model.weights):
  np.testing.assert_allclose(a.numpy(), b.numpy())