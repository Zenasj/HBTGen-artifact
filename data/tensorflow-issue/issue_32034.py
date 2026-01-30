import tensorflow as tf
from tensorflow.keras import layers

input1 = self.model.get_layer('input1').input
tf.summary.image('input1', input1 , max_outputs=MAX_OUT)

class MyLayer(keras.layers.Layer):
  ...
  def call(self, inputs):
    tf.summary.image(inputs)
    return 2*inputs

model = Model(...)
tb = keras.callbacks.TensorBoard(log_dir, update_freq=1)
model.fit(x, y, callbacks=[tb])