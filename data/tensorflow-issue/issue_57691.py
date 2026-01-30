import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import pathlib

def tflite_convert(model,data):
    def representative_data_gen():
            for input_value in data:
                input_value = input_value[np.newaxis, ...]
                yield [input_value] # shape should be (1, <data point size))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./output/tflite_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"model_{0}.tflite".format(model.name)
    tflite_model_file.write_bytes(tflite_model)

def tflite_explore(path):
    tflite_interpreter = tf.lite.Interpreter(model_path=path)
    tflite_interpreter.allocate_tensors()
    
    '''
    Check input/output details
    '''
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])
    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])


def make_dense_model():
    input_layer = tf.keras.Input(shape=(1,5,12))
    dense_layer=tf.keras.layers.Dense(3)(input_layer)
    model=tf.keras.Model(input_layer,dense_layer)
    return model

def generate_Noise_Data(shape,batch_size):
    if None in shape:
        shape=list(shape)
        shape[0]=batch_size  
    noise=np.array(np.random.randint(0,255,shape).astype(np.float32))
    return noise/255

model=make_dense_model()
print(model.summary())
in_value=generate_Noise_Data(model.layers[0].input_shape[0],2)
print(model(in_value))
model.save("./output/dense1layer.h5")
tflite_convert(model,in_value)
tflite_explore("./output/tflite_models/model_{0}.tflite".format(model.name))