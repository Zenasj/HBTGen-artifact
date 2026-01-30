from tensorflow.keras import models

# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import os
import random
from glob import glob
from PIL import Image
import numpy as np
import pathlib
import cv2
import onnx
from onnx_tf.backend import prepare
base_dir = "/media/sf_C_DRIVE/Users/ML/Desktop/blindbot_main"
modelpath = "/media/sf_C_DRIVE/Users/ML/Desktop/combined_model.onnx"
modelpath_keras = "/media/sf_C_DRIVE/Users/ML/Desktop/combined_model.hdf5"

class Data_Generator(Sequence):

    def __init__(self, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
      batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
      image = (np.array([ np.array(Image.open(file_name).resize(size=(320, 1024)), np.float32) for file_name in batch_x])/255)
      image = np.transpose(image, (0, 3, 2, 1))
      return image

validation_filenames = glob(base_dir + "/**/*.jpg",recursive=True)
random.shuffle(validation_filenames)
val_generator = Data_Generator(image_filenames=validation_filenames, batch_size=1)



def representative_data_gen():
  for i in range(1000):
    imgs = val_generator.__getitem__(i)
    yield [imgs[0:1]]

converter = tf.lite.TFLiteConverter.from_saved_model("/media/sf_C_DRIVE/Users/ML/Desktop/blindbot_main/tfmodel.pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8       # try to implement the model in tensorflow / try with int8 / try release candidate version for tensorflow / try model without concatenation layers / open issue on github of tensorflow / try monodepth tensorflow version
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_data_gen
converter.inference_type = tf.int8

tflite_models_dir = pathlib.Path("/media/sf_ML/Models/tflite_quantisized_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_quant = converter.convert()
tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)
print("done!")

def representative_data_gen():
   for i in range(1000):
       imgs = val_generator.getitem(i)
       yield [imgs[0:1].astype(np.int8)] # will yielding a single image will be possible here?

converter = tf.lite.TFLiteConverter.from_saved_model("/media/sf_C_DRIVE/Users/ML/Desktop/blindbot_main/tfmodel.pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS   #  tf.lite.OpsSet.TFLITE_BUILTINS_INT8 does not work ,switch to only built ins
]
converter.inference_input_type = np.int8 # try release candidate 
converter.inference_output_type = np.int8
converter.representative_dataset = representative_data_gen

converter._experimental_lower_tensor_list_ops = False