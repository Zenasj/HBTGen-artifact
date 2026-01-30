import random
from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from gtda.time_series import SlidingWindow
import matplotlib.pyplot as plt
from math import atan2, pi, sqrt

from tensorflow.keras.layers import Dense, MaxPooling1D, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tcn import TCN, tcn_full_summary
from tensorflow.keras.models import load_model

model = load_model('best_joint_new.hdf5',custom_objects={'TCN':TCN})
converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_no_quant_tflite = converter.convert()
open('best_joint.tflite', "wb").write(model_no_quant_tflite)

def representative_dataset():
    for i in range(trainX.shape[0]):
        yield ([trainX[i]])

# Set the optimization flag.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Enforce integer only quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
# Provide a representative dataset to ensure we quantize correctly.
converter.representative_dataset = representative_dataset
model_quant_tflite = converter.convert()
# Save the model to disk
open('best_joint_quant.tflite', "wb").write(model_quant_tflite)

trainX = numpy.random.rand(500,200,6)

import numpy
def representative_dataset():
    for i in range(500):
        yield ([numpy.random.rand(1,200,6).astype(np.float32)])