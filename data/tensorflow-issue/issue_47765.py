from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *

def get_student_model():
    base_model = MobileNetV2(weights=None, include_top=False,
                input_shape=(224, 224, 3))
    base_model.trainable = True
    inputs = Input(shape=(224, 224, 3)) 
    x = experimental.preprocessing.Rescaling(1./127.5, offset=-1)(inputs)
    y = base_model(x, training=True)
    y = GlobalAveragePooling2D()(y)
    y = Dense(512, activation="relu")(y)
    y = Dropout(0.5)(y)
    outputs = Dense(30, activation='softmax')(y)
    model = tf.keras.Model(inputs, outputs)
    return model

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_student_no_true_labels_e_125_t_2")
converter.experimental_new_converter = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
open("student_mobilenetv2.tflite", 'wb').write(tflite_model)
print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))

converter = tf.lite.TFLiteConverter.from_saved_model("mobilenetv2_student_no_true_labels_e_125_t_2")
converter._enable_tflite_resource_variables = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("student_mobilenetv2.tflite", 'wb').write(tflite_model)
print('Model size is %f MBs.' % (len(tflite_model) / 1024 / 1024.0))