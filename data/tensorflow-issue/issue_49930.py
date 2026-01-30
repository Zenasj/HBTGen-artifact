import random
from tensorflow import keras

import numpy as np
import tensorflow as tf

model = tf.keras.applications.EfficientNetB0(weights="imagenet")
norm_layer = model.layers[2]
assert "normalization" in norm_layer.name
print(norm_layer.mean.numpy())  # [0.485 0.456 0.406]
print(norm_layer.variance.numpy())  # [0.229 0.224 0.225]
# Generate sample inputs.
tf.random.set_seed(42)
x = tf.random.uniform((1, 224, 224, 3), 0, 255, dtype="int32", seed=42)


def get_reference(inputs):
    """Get the reference normalized outputs."""
    x = np.asarray(inputs).astype("float32")
    x /= 255.0
    x[..., 0] -= 0.485
    x[..., 1] -= 0.456
    x[..., 2] -= 0.406
    x[..., 0] /= 0.229
    x[..., 1] /= 0.224
    x[..., 2] /= 0.225
    return x


def get_current_tf_efficientnet_norm_output(inputs):
    """Get the normalized outputs from the current implementation."""
    x = np.asarray(inputs).astype("float32")
    x /= 255.0
    x[..., 0] -= 0.485
    x[..., 1] -= 0.456
    x[..., 2] -= 0.406
    x[..., 0] /= np.sqrt(0.229)
    x[..., 1] /= np.sqrt(0.224)
    x[..., 2] /= np.sqrt(0.225)
    return x


model_normalizer = tf.keras.Model(model.input, norm_layer.output)
# Below is True (they are the same)
np.allclose(get_current_tf_efficientnet_norm_output(x), model_normalizer(x), atol=1e-07)
# Below is False (they are different)
np.allclose(get_reference(x), model_normalizer(x), atol=1e-07)