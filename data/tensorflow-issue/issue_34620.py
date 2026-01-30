import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x=tf.keras.Input(shape=(256,1), batch_size=50)
y=tf.keras.layers.Convolution1D(kernel_size=5,filters=8,input_shape=(50,256,1))(x)
model=tf.keras.Model(x,y)
# compile and train
# ... 
converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = setgen
modelq=converter.convert()

conv_interpreter=tf.lite.Interpreter(model_content=modelq)
input_details = conv_interpreter.get_input_details()