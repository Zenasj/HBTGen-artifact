import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# create model
def get_model(input_shape=(None, None, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
            16, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.Conv2D(
            16 * 2, kernel_size=3, padding='same', strides=2)(x)
    x = tf.keras.layers.Conv2D(
            16 * 4, kernel_size=3, use_bias=False)(x)
    x = tf.keras.layers.Conv2D(
            3,  kernel_size=3, padding='same')(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error")
    return model


# Util functions
def save_tflite_model(output_model_path, tflite_model):
    with open(output_model_path, 'wb') as f:
        f.write(tflite_model)
    
def convert_model_from_concrete(model_path, output_model_path, input_shape=(1, None, None, 3)):
    model = tf.saved_model.load(model_path)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    
    concrete_func.inputs[0].set_shape(input_shape)
    
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS 
        ]
    tflite_model = converter.convert()
    print(tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True))
    save_tflite_model(output_model_path, tflite_model)


#Code for exporting my model to TFLite
model = get_model()
model.save("my_model_dynamic")
convert_model_from_concrete("my_model_dynamic","my_model_dynamic.tflite")

interpreter = tf.lite.Interpreter("my_model_dynamic.tflite")
custom_shape = [1, 512, 512, 3]
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], custom_shape)
interpreter.allocate_tensors()
input =  numpy.random.rand(*custom_shape).astype(np.float32)
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input)
interpreter.invoke()