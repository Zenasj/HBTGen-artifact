from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

input = tf.keras.Input(shape=[1], batch_size=1)
output = tf.keras.layers.Dense(
    10,
    kernel_initializer=tf.keras.initializers.Constant(value=0),
    bias_initializer=tf.keras.initializers.Constant(value=1),
)(input)
model = tf.keras.Model(inputs=[input], outputs=output)
model.summary()



def representative_dataset():
    yield [tf.constant([1.0])]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
converter._experimental_full_integer_quantization_bias_type = tf.int32 # Will work if commented

tflite_model = converter.convert()
with open("test.tflite", "wb") as fp:
    fp.write(tflite_model)



interpreter = tf.lite.Interpreter(
    model_path="test.tflite",
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF, # Will work if commented
)
interpreter.allocate_tensors()

input = [tf.constant([1.0])]
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input)
interpreter.invoke()
output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# [1.]
print(input)
# Expected: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
print(output)