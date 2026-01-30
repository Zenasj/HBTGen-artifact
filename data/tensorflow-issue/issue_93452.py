from tensorflow.keras import layers

import tensorflow as tf
import keras

threshold_keys = tf.constant(["a", "b"], dtype=tf.string, name="threshold_keys")
threshold_values = tf.constant([0.5, 0.7], dtype=tf.float32, name="threshold_values")
initializer = tf.lookup.KeyValueTensorInitializer(threshold_keys, threshold_values)
lookup_table = tf.lookup.StaticHashTable(initializer, default_value=0.0)

model = keras.layers.Dense(1)
model(tf.constant([[0.5]]))

export_archive = keras.export.ExportArchive()
model_fn = export_archive.track_and_add_endpoint(
    "model_fn",
    model, 
    input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)]
)

export_archive.track(lookup_table)

@tf.function()
def serving_fn(x):
    x = lookup_table.lookup(x)
    return model_fn(x)

x = tf.constant([["a"]])
serving_fn(x)
export_archive.add_endpoint(name="serve", fn=serving_fn)

export_archive.write_out("larifari") #<--- here i get the exception