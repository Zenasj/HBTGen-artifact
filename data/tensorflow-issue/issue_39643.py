import tensorflow as tf

class CustomLayer(Layer):
  def call(self, inputs):
    assert_same_batch_size = tf.assert_equal(tf.shape(inputs[0])[0],
                                                 tf.shape(inputs[1])[0], message="inputs do not have equal batch_size")
    with tf.control_dependencies([assert_same_batch_size]):
      return tf.reshape(inputs[0], (tf.shape(inputs[0])[0], 3))   # i want it to break

a = Input(shape=1)
b = Input(shape=1)
out = CustomLayer()([a, b])
m = Model([a, b], out)

assert_same_batch_size = tf.assert_equal(tf.shape(inputs[0]),
                                                 tf.shape(inputs[1]), message="inputs do not have equal batch_size")