import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs=tf.keras.layers.Input([6,6,1280])
x=tf.keras.layers.GlobalAveragePooling2D(
    name='global_average_pooling2d_0')(inputs)
y=tf.keras.layers.Dense(5,name='dense_0')(x)
model_0 = tf.keras.Model(inputs,y,name='model_0')

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192,192,3),
    include_top=False,
    weights='imagenet')
assert mobile_net.output_shape==(None, 6, 6, 1280)
y1=model_0.get_layer('global_average_pooling2d_0')(mobile_net.output)
y2=model_0.get_layer('dense_0')(y1)
model_1 = tf.keras.Model(mobile_net.input,y2,name='assembled_model')


tmp=tf.keras.Model(model_1.input,model_1.layers[-3].output) # successfully
tmp=tf.keras.Model(model_1.input,model_1.layers[-2].output) # failed
tmp=tf.keras.Model(model_1.input,model_1.layers[-1].output) # failed