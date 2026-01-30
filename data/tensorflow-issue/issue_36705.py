import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import keras
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np


n = 16

dense_1 = keras.layers.Dense(n)
dense_2 = keras.layers.Dense(n)

x = keras.layers.Input(shape=(n,))

y = dense_1(x)
y = dense_2(y)

model = keras.models.Model(inputs=[x], outputs=[y])

model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.save('example.h5')

input_shapes = {}
input_shapes['input_1'] = (1, n)
converter = tf.lite.TFLiteConverter.from_keras_model_file('example.h5', input_shapes=input_shapes)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

def rep_data_gen():
	for _ in range(100):
		yield [np.random.rand(1, n).astype(np.float32)]

converter.representative_dataset = rep_data_gen

tflite_model = converter.convert()
open("example.tflite", "wb").write(tflite_model)

interpreter = tflite.Interpreter("example.tflite",
								 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

y = dense_2(y)

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

x = keras.layers.Input(shape=(n,))

import tensorflow as tf
if not str(tf.__version__).startswith('1.15'):
    print('please use tensorflow 1.15')
    exit()
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense

tf.enable_eager_execution()

n = 64

# image_shape = (n,n,3)
image_shape = (n,)

# Creating a dummy keras model here
x = Input(shape=image_shape)

# y = Conv2D(3, (3, 3), padding='same')(x)
# y = Conv2D(3, (3, 3), padding='same')(y)

y = Dense(n)(x)
y = Dense(n)(y)

model = Model(inputs=x, outputs=y)
model.summary()
model.save('keras_model.h5', include_optimizer=False)

def representative_dataset_gen():
    for i in range(100):
        # creating fake images
        image = tf.random.normal([1] + list(image_shape))
        yield [image]

# actual conversion
converter = tf.lite.TFLiteConverter.from_keras_model_file('keras_model.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # For EdgeTPU, no float ops allowed
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# save model
tflite_model = converter.convert()
open('dummy.tflite', 'wb').write(tflite_model)

# How to Reproduce issue:
# 1) Install TensorFlow 1.15 ('1.15.0') and edgetpu_compiler
# 2) python simple_example.py (this file)
# 3) edgetpu_compiler issue_36705_dummy.tflite
#    - Will fail with ERROR: :129 std::abs(input_product_scale - bias_scale) <= 1e-6 * std::min(input_product_scale, bias_scale) was not true.
# 4) Set 'will_work = True' below:
# 5) Repeat steps 2 & 3 and it will compile successfully.

import tensorflow as tf
import numpy as np
import pdb 


def main():
  tflite_filepath = "issue_36705_dummy.tflite"
  will_work = False  # Switch to trigger bug/fix
  
  # Hyperparameters
  tflite_batch_size = 1 
  input_size = 64

  num_layers = 2 
  default_layer_size = 1024
  if will_work:
    # Make the layers different sizes to avoid the issue.
    layer_sizes = [default_layer_size - i for i in range(num_layers)]
  else:
    layer_sizes = [default_layer_size] * num_layers
  
  # Build Graph
  input_shape = (tflite_batch_size, input_size)
  x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)

  output = x 
  for i in range(num_layers):
    # Can use tf.matmul or tf.keras.layers.Dense here too:
    output = tf.contrib.layers.fully_connected(output, layer_sizes[i])

  # Build TFLite
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    converter = tf.compat.v1.lite.TFLiteConverter.from_session(sess, 
        [x], [output])

    def representative_dataset_gen():
      for i in range(100):
        yield [np.ones(input_shape).astype(np.float32)]

    # UINT8 + Edge TPU Constraints.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Build and write binary.
    tflite_binary = converter.convert()
    open(tflite_filepath, "wb").write(tflite_binary)


if __name__ == "__main__":
  main()