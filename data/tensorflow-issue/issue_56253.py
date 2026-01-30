import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
class CFGGG:
    SAVED_MODEL_DIR='./zz_tflite_op_support/demo'
    TFLITE_MODEL_FILENAME='./zz_tflite_op_support/demo.tflite'

inputs=tf.keras.layers.Input((5,5,2))
def pre_process(x):
    grid_x,grid_y=tf.shape(x)[1:3]
    # range_x=tf.range(5) # when replace 5 with grid_x，runtime error occurs when invoke
    # range_y=tf.range(5) # when replace 5 with grid_y，runtime error occurs when invoke
    range_x=tf.range(grid_x) 
    range_y=tf.range(grid_y) 
    x_grid,y_grid=tf.meshgrid(range_y,range_x)
    
    b=tf.stack([y_grid,x_grid],axis=-1)
    b=tf.cast(b,tf.float32)
    y=x+b
    return y
    
x=pre_process(inputs)
outputs=tf.keras.layers.Dense(10)(x)

model_debug=tf.keras.Model(inputs=inputs,outputs=x)
model_debug.summary()


model_debug.save(CFGGG.SAVED_MODEL_DIR)

converter = tf.lite.TFLiteConverter.from_saved_model(CFGGG.SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True


tflite_model = converter.convert()

with open(CFGGG.TFLITE_MODEL_FILENAME, 'wb') as f:
    f.write(tflite_model)
    
interpreter = tf.lite.Interpreter(model_path=CFGGG.TFLITE_MODEL_FILENAME)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Explore the model input output shape.
print(input_details)
print(output_details)

input_shape = input_details[0]['shape']
print("input shape",input_shape)
for i, output_detail in enumerate(output_details):
    output_shape = output_detail['shape']
    print("No {} output shape {}".format(i,output_shape))


# Test the model on random input data.
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
print("input data complete: ",input_data.shape)
interpreter.invoke()
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
for i in range(len(output_details)):
    output_data_tmp = interpreter.get_tensor(output_details[i]['index'])
    output_shape_tmp = output_data_tmp.shape
    print("invoke() complete，No {} output shape {}".format(i,output_shape_tmp))

def pre_process(x):
    
    grid_x,grid_y=tf.shape(x)[1:3]
    x_grid=tf.ones([grid_x,grid_y])
    y_grid=tf.ones([grid_x,grid_y])
    b=tf.stack([x_grid,y_grid],axis=-1)
    y=x+tf.cast(b,tf.float32)
    
    return y

import tensorflow as tf
import numpy as np
class CFGGG:
    SAVED_MODEL_DIR='./zz_tflite_op_support/demo'
    TFLITE_MODEL_FILENAME='./zz_tflite_op_support/demo.tflite'

inputs=tf.keras.layers.Input((5,5,2))
def pre_process_shows_shape_OK(x):
    grid_x,grid_y=tf.shape(x)[1:3]
    x_grid=tf.ones([grid_x,grid_y])
    y_grid=tf.ones([grid_x,grid_y])
    b=tf.stack([x_grid,y_grid],axis=-1)
    y=x+tf.cast(b,tf.float32)
    return y
def pre_process_shows_range_meshgrid_OK(x):
    range_x=tf.range(5) #  bug appears  when replace with grid_x 替换为grid_x，tflite在invoke的时候就出错
    range_y=tf.range(5) # bug appears  when replace with grid_x 替换为grid_y，tflite在invoke的时候就出错
    x_grid,y_grid=tf.meshgrid(range_y,range_x)
    
    b=tf.stack([y_grid,x_grid],axis=-1)
    b=tf.cast(b,tf.float32)
    y=x+b
    return y
def pre_process_shows_shape_range_meshgrid_BUG(x):
    grid_x,grid_y=tf.shape(x)[1:3]
    grid_x=tf.shape(x)[1]
    grid_y=tf.shape(x)[2]
    range_x=tf.range(grid_x) # bug appears  替换为grid_x，tflite在invoke的时候就出错
    range_y=tf.range(grid_y) # bug appears   替换为grid_y，tflite在invoke的时候就出错
    x_grid,y_grid=tf.meshgrid(range_y,range_x)
    
    b=tf.stack([y_grid,x_grid],axis=-1)
    b=tf.cast(b,tf.float32)
    y=x+b
    return y
def pre_process_shows_shape_range_OK(x):
    grid_x=tf.shape(x)[1]
    grid_y=tf.shape(x)[2]
    range_x=tf.range(grid_x) 
    range_y=tf.range(grid_y) 
    b=tf.stack([range_x,range_y],axis=-1)
    return b

# x=pre_process_shows_shape_OK(inputs)
# x=pre_process_shows_range_meshgrid_OK(inputs)
# x=pre_process_shows_shape_range_OK(inputs)
x=pre_process_shows_shape_range_meshgrid_BUG(inputs)

outputs=tf.keras.layers.Dense(10)(x)

model_debug=tf.keras.Model(inputs=inputs,outputs=x)
model_debug.summary()

model_debug.save(CFGGG.SAVED_MODEL_DIR)

# then conver to tflite, then read tflite, then  interpreter.invoke()
converter = tf.lite.TFLiteConverter.from_saved_model(CFGGG.SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True


tflite_model = converter.convert()

with open(CFGGG.TFLITE_MODEL_FILENAME, 'wb') as f:
    f.write(tflite_model)
    
interpreter = tf.lite.Interpreter(model_path=CFGGG.TFLITE_MODEL_FILENAME)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Explore the model input output shape.
print(input_details)
print(output_details)

input_shape = input_details[0]['shape']
print("input shape",input_shape)
for i, output_detail in enumerate(output_details):
    output_shape = output_detail['shape']
    print("No {} output shape {}".format(i,output_shape))


# Test the model on random input data.
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
print("input data complete: ",input_data.shape)
interpreter.invoke()
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
for i in range(len(output_details)):
    output_data_tmp = interpreter.get_tensor(output_details[i]['index'])
    output_shape_tmp = output_data_tmp.shape
    print("invoke() complete，No {} output shape {}".format(i,output_shape_tmp))

tf.__version__
Out[7]: '2.8.0'

def pre_process_shows_shape_range_meshgrid_BUG(x):
    grid_x,grid_y=tf.shape(x)[1:3]
    grid_x=tf.shape(x)[1]
    grid_y=tf.shape(x)[2]
    range_x=tf.range(grid_x) # bug dis-appears  
    range_y=tf.range(grid_y) # bug dis-appears  
    x_grid,y_grid=tf.meshgrid(range_y,range_x)
    
    b=tf.stack([y_grid,x_grid],axis=-1)
    b=tf.cast(b,tf.float32)
    y=x+b
    return y

converter = tf.lite.TFLiteConverter.from_saved_model(CFGGG.SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

converter = tf.lite.TFLiteConverter.from_saved_model(CFGGG.SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True