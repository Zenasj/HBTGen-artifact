from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

# Mock a model
input_x = tf.keras.Input(1)
output_y = tf.keras.layers.Dense(1)(input_x)
model = tf.keras.Model(input_x, output_y)

def _get_serving_signature(model):
    @tf.function
    def my_func(x, y):
        x_out = tf.cast(tf.sparse.to_indicator(x, 5), tf.int64)
        return {"x": x_out, "y": y}
    
    return my_func

# Get the concrete func
concrete_func = _get_serving_signature(model).get_concrete_function(
        x=tf.SparseTensorSpec(shape=[None, None], dtype=tf.int64),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
    )

# Store this function inside the model
model.my_func = concrete_func
    
# Build the signature dict
signatures = { "default": concrete_func}

# save the model
model.save("./serving_dir", save_format="tf", signatures=signatures)

# Load the model back
model2 = tf.keras.models.load_model("./serving_dir")

# Make up some data
x = tf.ragged.constant([[1, 3], [2, 3, 1], [2]], dtype=tf.int64).to_sparse()
y = tf.expand_dims(tf.constant([1, 2, 1], dtype=tf.int64), axis=1)

# Make inference on saved my_func
out_func_attr = model2.my_func(x=x, y=y)

# Make inference using signature
out_signature = model2.signatures["default"](x, y)

out_func_attr = model2.my_func(x=x, y=y)

out_signature = model2.signatures["default"](x, y)