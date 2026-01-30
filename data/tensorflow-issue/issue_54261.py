from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Reshape

inputs = Input(shape=(10,))
MLP = Sequential()
MLP.add(Dense(1024*5))
MLP.add(Reshape((5,1,1,1024)))
MLP.add(BatchNormalization(axis=1))
MLP.add(Dense(1024*5))

output = MLP(inputs)

model = Model(inputs=inputs, outputs=output)
model.save('reproduce_bug')
converter = tf.lite.TFLiteConverter.from_saved_model('reproduce_bug')

converter._experimental_lower_tensor_list_ops = False
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

tflite_model = converter.convert()

with open("reproduce_bug.tflite", "wb") as f:
    f.write(tflite_model)

import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='reproduce_bug.tflite')

for tdetails in interpreter.get_tensor_details():
    if tdetails['index'] == 19: # or your failing index
        print(tdetails)

interpreter.allocate_tensors()