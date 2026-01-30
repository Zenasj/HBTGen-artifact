from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

def _crop_and_concat(inputs, residual_input):
  factor = inputs.get_shape().dims[1].value / residual_input.get_shape().dims[1].value
  return tf.concat([inputs, tf.image.central_crop(residual_input, factor)], axis=-1)

class UNet(tf.keras.Model):
  def __init__(self, name):
    super(UNet, self).__init__(name)
    self.conv1 = tf.keras.layers.Conv2D(filters=8,
                                        kernel_size=(3, 3),
                                        activation=tf.nn.relu)
    self.conv2 = tf.keras.layers.Conv2D(filters=8,
                                        kernel_size=(3, 3),
                                        activation=tf.nn.relu)
    self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                             strides=2)
    self.deconv = tf.keras.layers.Conv2DTranspose(filters=16,
                                                  kernel_size=(2, 2),
                                                  strides=(2, 2),
                                                  padding='same',
                                                  activation=tf.nn.relu)
    self.conv3 = tf.keras.layers.Conv2D(filters=8,
                                        kernel_size=(3, 3),
                                        activation=tf.nn.relu)

  @tf.function
  def call(self, x):
    print(">>> Input Shape", x.shape)
    out = self.conv1(x)
    print(">>> conv1 Shape", out.shape)
    skip = self.conv2(out)
    print(">>> conv2 Shape", skip.shape)
    out = self.maxpool(skip)
    print(">>> maxpool Shape", out.shape)
    out = self.deconv(out)
    # the deconv shape will be (None, None, None, 16) when exporting saved model
    print(">>> deconv Shape", out.shape)
    out = self.conv3(out)
    out = _crop_and_concat(out, skip)

    return out


model = UNet("dummy")

res = model.predict(tf.ones((1, 400, 400, 1)))

print("Finish prediction")

tf.keras.models.save_model(model, "/results/SavedModel",
    save_format="tf", overwrite=True, include_optimizer=False)