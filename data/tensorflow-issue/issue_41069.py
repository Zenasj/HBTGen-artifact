from tensorflow import keras
from tensorflow.keras import models

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(200,200,3), dropout=.2)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512,activation='relu')(x)
types = Dense(20,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=types)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(200,200,3), dropout=.2)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512,activation='relu')(x)
types = Dense(20,activation='softmax')(x)

y = base_model.output
y = GlobalAveragePooling2D()(y)
y = Dense(512, activation='relu')(y)
y = Dense(1024, activation='relu')(y)
y = Dense(512, activation='relu')(y)
values = Dense(3, activation='sigmoid')(y)

model = Model(inputs=base_model.input, outputs=[types,values])

interpreter = tf.lite.Interpreter(model_path='path/to/model/quantized.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("shape:", output_details[0]['shape'])
print("shape:", output_details[1]['shape'])

#reshape for input of batch size 32
interpreter.resize_tensor_input(input_details[0]['index'], (32, 200, 200, 3))
interpreter.resize_tensor_input(output_details[0]['index'], (32, 20))
#interpreter.resize_tensor_input(output_details[1]['index'], (32, 3))
interpreter.allocate_tensors()

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet import preprocess_input
import pathlib

model = tf.keras.models.load_model('path/to/file.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

os.chdir('/directory/of/image/directories')#Where image directories are
directory = os.listdir()
directory

def representative_dataset_gen():
    for i in directory:
        count = 0
        os.chdir(i)
        files = os.listdir()
        print(i)
        for j in files:
            if count<600:
                img = Image.open(j)
                array = np.asarray(img, dtype=np.float32)
                array = preprocess_input(array)
                count=count+1
                yield[np.expand_dims(array, axis=0)]
            else:
                break      
        os.chdir('../')

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

#Saving file
tflite_model_dir = pathlib.Path('save/path/')
tflite_quant_model_file = tflite_model_dir/'quantized.tflite'
tflite_quant_model_file.write_bytes(tflite_quant_model)