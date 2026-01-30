from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# Load Pre-Trained Model
model = ...
# Using I/O of previous model, build model with an additional (or replacement) output 
first_half = tf.keras.models.Model(model.inputs, model.get_layer("some_intermediate_layer_n").output)

second_half = tf.keras.models.Model(model.get_layer("some_intermediate_layer_n_plus_1").input, model.outputs)

import tensorflow as tf
def def_enc_dec_pair():
    input_ = tf.keras.layers.Input(shape=[None, None, 3])
    x_e = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(input_)
    x_e = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x_e)
    x_e = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x_e)
    x_e = tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="encoded_layer")(x_e)

    x_d = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="second_section")(x_e)
    x_d = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x_d)
    x_d = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x_d)
    x_d = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x_d)

    model = tf.keras.Model(inputs=[input_], outputs=[x_d])
    model.compile(optimizer=tf.optimizers.Adam(), loss=[tf.keras.losses.MSE])

    return model

base_model = def_enc_dec_pair()
# base_model.summary()

first_half = tf.keras.Model(base_model.inputs, base_model.get_layer("encoded_layer").output)
# first_half.summary()

second_half = tf.keras.Model(base_model.get_layer("second_section").input, base_model.output)
second_half.summary()

...
second_input_shape = base_model.get_layer("encoded_layer").output.shape
second_input_shape = second_input_shape[1:]  # Remove Batch from front.
second_input = tf.keras.layers.Input(shape=second_input_shape)
x = second_input
for layer in base_model.layers[5:]:  # encoded_layer idx + 1
    x = layer(x)

second_half = tf.keras.Model(second_input, x)
second_half.summary()

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder_layers = []
    self.encoder_layers.append(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
    self.encoder_layers.append(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
    self.encoder_layers.append(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
    self.encoder_layers.append(tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="encoded_layer"))

  def call(self, inputs):
    output = inputs
    for layer in self.encoder_layers:
      output = layer(output)
    return output

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    self.decoder_layers = []
    self.decoder_layers.append(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME", name="second_section"))
    self.decoder_layers.append(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
    self.decoder_layers.append(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
    self.decoder_layers.append(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME"))
  
  def call(self, inputs):
    output = inputs
    for layer in self.decoder_layers:
      output = layer(output)
    return output


class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def call(self, inputs):
    return self.decoder(self.encoder(inputs))