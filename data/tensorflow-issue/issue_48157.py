from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def build_model():
    input_tensor = tf.keras.layers.Input((32, 32, 128))
    x = input_tensor

    boxes = tf.keras.layers.Conv2D(4, (3, 3))(x)
    boxes = tf.reshape(boxes, (-1, 4))

    scores = tf.keras.layers.Conv2D(1, (3, 3))(x)
    scores = tf.reshape(scores, (-1,))

    return  tf.keras.Model({"inputs": input_tensor}, {"boxes": boxes, "scores": scores})

model = build_model()
model = build_model()

model.save("/tmp/model")

restored_fn = tf.saved_model.load("/tmp/model")
concrete_fn = restored_fn.signatures["serving_default"]

# Would be great if these would match the Keras model:
# signature_wrapper(*, input_6)
# Args:
#   input: float32 Tensor, shape=(None, 32, 32, 128)
# Returns:
#   {'boxes': <1>, 'scores': <2>}
#     <1>: float32 Tensor, shape=(None, 4)
#     <2>: float32 Tensor, shape=(None,)

# Instead the keys are layer names which depends on implementation details and on how often the model
# was instantiated: (e.g input_6, tf.reshape_8, tf.reshape_9)
print(concrete_fn.pretty_printed_signature())

@tf.function(input_signature=[tf.TensorSpec([None, 32, 32, 128], dtype=tf.float32, name="input")])
def override_output_signatures(input_tensor):
   outputs = model(input_tensor)
   return {"boxes": outputs["boxes"], "scores": outputs["scores"]}

signatures = override_output_signatures.get_concrete_function()
tf.saved_model.save(model, export_dir="/tmp/saved_model", signatures=signatures)

from collections import OrderedDict
import tensorflow as tf

def build_model():
    input_tensor = tf.keras.layers.Input((32, 32, 128))
    x = input_tensor

    scores = tf.keras.layers.Conv2D(1, (3, 3))(x)
    scores = tf.reshape(scores, (-1,))

    boxes = tf.keras.layers.Conv2D(4, (3, 3))(x)
    boxes = tf.reshape(boxes, (-1, 4))

    return tf.keras.Model({"inputs": input_tensor}, {"scores": scores, "boxes": boxes})

model = build_model()
model.save("/tmp/model")
restored_fn = tf.saved_model.load("/tmp/model")
concrete_fn = restored_fn.signatures["serving_default"]
print(concrete_fn.pretty_printed_signature())

@tf.function(input_signature=[tf.TensorSpec([None, 32, 32, 128], dtype=tf.float32, name="input")])
def override_output_signatures(input_tensor):
   outputs = model(input_tensor)
   return OrderedDict([("scores", outputs["scores"]), ("boxes", outputs["boxes"])])

signatures = override_output_signatures.get_concrete_function()
tf.saved_model.save(model, export_dir="/tmp/saved_model", signatures=signatures)
print(tf.saved_model.load("/tmp/saved_model").signatures["serving_default"].pretty_printed_signature())