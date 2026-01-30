import math
import random
from tensorflow import keras

import tensorflow as tf
import tflite
import numpy as np
import imutils

tf.compat.v1.enable_eager_execution()

class MyFunc(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32, name='imdata')])
    def detect(self, imdata):
        # both tf.keras.backend.var and tf.math.reduce_variance operations create the same error scores
        #tensor_var = tf.keras.backend.var(imdata)
        tensor_var = tf.math.reduce_variance(imdata)
        return tensor_var
tf_func = MyFunc()

conc_func = tf_func.detect.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([conc_func])
fpath = 'model.tflite'
with tf.io.gfile.GFile(fpath, 'wb') as f:
    f.write(converter.convert())
loaded_model = tf.lite.Interpreter(fpath)

def tflite_inference(tmp_model, tmp_img_data):
    input_details = tmp_model.get_input_details()
    output_details = tmp_model.get_output_details()
    input_data = np.array(tmp_img_data).astype(input_details[0]['dtype'])
    w_img, h_img, ch_img = tmp_img_data.shape
    tmp_model.resize_tensor_input(input_details[0]["index"], [w_img, h_img, 3])
    tmp_model.allocate_tensors()
    tmp_model.set_tensor(input_details[0]['index'], tmp_img_data)
    tmp_model.invoke()
    result = tmp_model.get_tensor(output_details[0]['index'])
    return result


images = ['https://images.unsplash.com/photo-1470020337050-543c4e581988',
          'https://images.unsplash.com/photo-1516811108838-030371f93644',
          'https://zipbooks.com/wp-content/uploads/2017/05/royalty-free-images-free-of-charge.jpeg',
       ]

print("tensorflow vs tflite inference scores on images")
for img_url in images:
    tmp_imdata = imutils.url_to_image(img_url).astype(np.float32)
    print(f'\t tensorflow: {tf_func.detect(tmp_imdata).numpy():.4f}', end='\t')
    print(f'tflite: {tflite_inference(loaded_model, tmp_imdata):.4f}')

    
print("\n\n tensorflow vs tflite inference scores on random arrays")
shape = (2000, 2000, 3)
num_range = 255
arrays = [np.random.randint(num_range, size=shape).astype(np.float32), 
          np.random.randint(num_range, size=shape).astype(np.float32),
          np.random.randint(num_range, size=shape).astype(np.float32)
         ]
print('\t tensorflow \t tflite \t   Equal?')
for arr in arrays:
    tf_score = tf_func.detect(arr).numpy()
    tflite_score = tflite_inference(loaded_model, arr)
    eq = tf_score == tflite_score
    print(f'\t {tf_score:.4f} \t {tflite_score:.4f} \t   {eq}')

class MyFunc(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32, name='imdata')])
    def detect(self, imdata):
        tensor_var = tf.math.reduce_sum(imdata)
        return tensor_var
tf_func = MyFunc()

print("\n\n tensorflow vs tflite inference scores on random arrays")
shape = (200, 200, 3)
num_range = 255
arrays = [np.random.randint(num_range, size=shape).astype(np.float32), 
          np.random.randint(num_range, size=shape).astype(np.float32),
          np.random.randint(num_range, size=shape).astype(np.float32)
         ]
print('\t tensorflow \t tflite \t   Equal?')
for arr in arrays:
    tf_score = tf_func.detect(arr).numpy()
    tflite_score = tflite_inference(loaded_model, arr)
    eq = tf_score == tflite_score
    print(f'\t {tf_score:.4f} \t {tflite_score:.4f} \t   {eq}')