import random
from tensorflow.keras import layers
from tensorflow.keras import models

# The full neural network code!
###############################
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import UpSampling1D

#Reference:https://colab.research.google.com/gist/ohtaman/c1cf119c463fd94b0da50feea320ba1e/edgetpu-with-keras.ipynb
def UpSamplingCustom2D(scale=(2, 2)):
  if isinstance(scale, int):
    scale = (scale, scale)

  def upsampling(x):
    shape = x.shape
    print("Upsampling : ", shape)
    x = keras.layers.Concatenate(-2)([x] * scale[0])
    x = keras.layers.Reshape([shape[1] * scale[0], shape[2], shape[3]])(x)
    x = keras.layers.Concatenate(-1)([x] * scale[1])
    x = keras.layers.Reshape([shape[1] * scale[0], shape[2] * scale[1], shape[3]])(x)
    return x

  return upsampling


image_size = 28
output_image_size = image_size * 2

train_images = np.random.rand(10,image_size,image_size,3)
train_images_labels = np.random.rand(10,output_image_size,output_image_size,3)


test_images = np.random.rand(10,image_size,image_size,3)
test_images_labels = np.random.rand(10,output_image_size,output_image_size,3)

inputs = keras.layers.Input(shape=(image_size, image_size, 3))
x = keras.layers.Dense(image_size, activation='relu')(inputs)
x = keras.layers.Dense(image_size, activation='relu')(x)
x = UpSamplingCustom2D()(x)
decoded = keras.layers.Dense(3, activation='relu')(x)
model = keras.models.Model(inputs, decoded)

#Compile the model.
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
)

#Train the model.
model.fit(
  train_images,
  train_images_labels,
  epochs=1,
  batch_size=32,
)

#Evaluate the model.
model.evaluate(
  test_images,
  test_images_labels
)



#Save the model to disk.
model.save_weights('model.h5')

#Load the model from disk later using:
model.load_weights('model.h5')

#Predict on the first 5 test images.
predictions = model.predict(test_images)

print(predictions[0].shape)


def representative_dataset_gen():
  for i in range(5):
    yield [train_images[i: i + 1].astype(np.float32)]

keras_file = "upsampling2d.h5"
tf.keras.models.save_model(model, keras_file)

#Convert to TensorFlow Lite model.
if (tf.__version__ == '1.14.0'):
  converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
if (tf.__version__ == '2.0.0-beta1'):
  converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_fname = "upsampling2d_" + str(tf.__version__) + ".tflite"
open(tflite_fname, "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=tflite_fname)
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

print("Input Details : ", input_detail)
print("Output Details : ", output_detail)

def quantize(real_value):
  std, mean = input_detail['quantization']
  return (real_value / std + mean).astype(np.uint8)


sample_input = quantize(test_images[0]).reshape(input_detail['shape'])
print("Sample Input Shape : ", sample_input.shape)
print("Inference Sample Input Shape : ", sample_input.shape)

interpreter.set_tensor(input_detail['index'], sample_input)
interpreter.invoke()

#original_image = test_images[0].reshape((28, 28))
pred_original_model = model.predict(test_images[:1])
pred_quantized_model = interpreter.get_tensor(output_detail['index'])

print("Prediction : ",pred_quantized_model.shape)