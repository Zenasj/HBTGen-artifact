import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model_conv = efn.EfficientNetB0(weights='/work/source/pre_trained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',include_top=False,input_tensor=input_layer)
model = tf.keras.Sequential()
model.add(tf.keras.layers.TimeDistributed(model_conv, input_shape=(3, 1024,1024,3)))
model.add(tf.keras.layers.GlobalAveragePooling3D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(classes_n - 1))
model.add(tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions'))

input_layer = tf.keras.layers.Input(shape=(1024,1024,3))
model_conv = efn.EfficientNetB0(weights='/work/source/pre_trained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
                                include_top=False,input_tensor=input_layer)

model_conv = tf.keras.Model(model_conv.input,model_conv.get_layer('block6d_add').output)
model_conv = tf.keras.models.load_model('./models_cut/effb0_5block.h5')
model = tf.keras.Sequential()
model.add(tf.keras.layers.TimeDistributed(model_conv, input_shape=(3, 1024,1024,3)))
model.add(tf.keras.layers.GlobalAveragePooling3D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(classes_n - 1))
model.add(tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions'))