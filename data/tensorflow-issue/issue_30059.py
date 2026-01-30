from tensorflow import keras
from tensorflow.keras import models

grads = K.gradients(class_output, last_conv_layer.output)[0]

import numpy as np
import cv2
import tensorflow as tf
tf.enable_eager_execution()

model = tf.keras.models.load_model("model.h5")
print(type(model))
# tensorflow.python.keras.engine.sequential.Sequential

from dataset import prepare_dataset
_, ds, _, _, _, _ = prepare_dataset() # ds is a tf.data.Dataset
print(type(ds))
# tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter

it = train_ds.make_one_shot_iterator()
img, label = it.get_next()
print(type(img), img.shape)
# <class 'tensorflow.python.framework.ops.EagerTensor'> (192, 192, 3)

print(type(label), label.shape)
# <class 'tensorflow.python.framework.ops.EagerTensor'> (2,)

img = np.expand_dims(img, axis=0)
print(img.shape)
# (1, 192, 192, 3)

predictions = model.predict(img)
print(predictions)
# array([[0.9711799 , 0.02882008]], dtype=float32)

class_idx = np.argmax(predictions[0])
print(class_idx)
# 0

class_output = model.output[:, class_idx]
print(model.output, class_output)
# Tensor("Softmax:0", shape=(?, 2), dtype=float32) Tensor("strided_slice_5:0", dtype=float32)

last_conv_layer = model.get_layer('conv2d_33') # the last conv layer

"""
Now, the fun part: how do I compute the gradient of class_output with respect to
the output of the last convolutional layer?
"""

with tf.GradientTape() as tape: 
    print(label)
    # tf.Tensor([1. 0.], shape=(2,), dtype=float32)
    y_c = tf.reduce_sum(tf.multiply(model.output, label))
    print(y_c)
    # Tensor("Sum_4:0", shape=(), dtype=float32)
    last_conv_layer = model.get_layer('activation_6')

grad = tape.gradient(y_c, last_conv_layer.output)