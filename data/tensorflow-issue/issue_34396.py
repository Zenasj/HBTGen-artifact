from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_hub as hub
import tensorflow as tf 

max_seq_length = 128  # Your choice here.

input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer =  hub.KerasLayer("https://tfhub.dev/tensorflow/albert_lite_base/1",
                            signature="tokens",
                            output_key="pooled_output")

albert_inputs = dict(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)

pooled_output = bert_layer(albert_inputs)

model = tf.keras.Model(inputs=[input_ids, input_mask, segment_ids], outputs=[pooled_output])
model.compile()


converter = tf.lite.TFLiteConverter.from_keras_model(model)                                
tflite_model = converter.convert()

albert_module = hub.load("https://tfhub.dev/tensorflow/albert_lite_base/1")
# also works with https://tfhub.dev/google/albert_base/3 and https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/1

converter = tf.lite.TFLiteConverter.from_concrete_functions([albert_module.signatures["tokens"]])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with tf.io.gfile.GFile(os.path.join("../", "model.tflite"), 'wb') as f:
    f.write(tflite_model)

def load_module(module_url, signature='tokens'):
    """Load a module in tensorflow 2"""
    module = hub.load(module_url, tags=[])
    log.info(f"found signatures {module.signatures}")
    return module.signatures[signature]

embedder = load_module('https://tfhub.dev/google/albert_xlarge/2')