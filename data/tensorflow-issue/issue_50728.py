from tensorflow import keras
from tensorflow.keras import layers

if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(input_shape)
      outputs.set_shape(out_shape)

if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = Conv.compute_output_shape(self, input_shape)
      outputs.set_shape(out_shape)

import tensorflow as tf
print(tf.version.VERSION)

class MyFlatConv(tf.keras.layers.Conv1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def compute_output_shape(self, input_shape):

        res_shape = super().compute_output_shape(input_shape)
        return res_shape[0], res_shape[1]*res_shape[2]

    @tf.function
    def call(self, inputs):
        res = super().call(inputs)

        return tf.reshape(res, (res.shape[0], -1))

if __name__ == "__main__":

    mm = MyFlatConv(filters=3, kernel_size=3, padding="same")
    zz= tf.zeros((3,10,3), dtype=tf.float32)
    print(mm(zz))