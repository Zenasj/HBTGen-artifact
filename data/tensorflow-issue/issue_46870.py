from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

x = layers.Input(shape=(256,64,3))
y = layers.Dense(1,use_bias=False)(x) #here use_bias=False is the key point
y = layers.GlobalMaxPool2D()(y)

model = keras.Model(inputs=[x],outputs=[y])
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_input = model.inputs[0]
input_shape = model_input.shape
model_input.set_shape((1,*input_shape[1:]))

lite_model = converter.convert()
with open('issue_dense_2dMustBias_android_gpu.tflite', "wb") as fp:
    fp.write(lite_model)