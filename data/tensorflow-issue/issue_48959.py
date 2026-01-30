import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(tf.__version__)

class MyModel(tf.keras.Model):
    def __init__(self, squeeze=False):
        super().__init__()
        self.squeeze = squeeze
        self.dense = tf.keras.layers.Dense(5)
    
    def call(self, x):
        if self.squeeze:
            x = tf.squeeze(x)
        return self.dense(x)
    
def representative_dataset():
    for _ in range(100):
        yield [tf.random.uniform(shape=[1, 30, 5])]
   
model = MyModel(squeeze=False)
model(tf.random.uniform(shape=[1, 30, 5]))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

tflite_model = converter.convert()

input_ = tf.random.uniform(shape=[1, 30, 5])
output = model(input_)

interpreter = tf.lite.Interpreter(model_content=tflite_model)

interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_)
interpreter.invoke()
tflite_output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

print(tf.reduce_max(tf.abs(output - tflite_output)))

model = MyModel(squeeze=True)
model(tf.random.uniform(shape=[1, 30, 5]))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

tflite_model = converter.convert()

input_ = tf.random.uniform(shape=[1, 30, 5])
output = model(input_)

interpreter = tf.lite.Interpreter(model_content=tflite_model)

interpreter.allocate_tensors()
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_)
interpreter.invoke()
tflite_output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

print(tf.reduce_max(tf.abs(output - tflite_output)))