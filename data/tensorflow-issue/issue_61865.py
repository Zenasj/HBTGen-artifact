import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  @tf.function(jit_compile=True)
  def call(self, input):
    x = tf.gather(input, indices=[5, 8, 7, 16, 256, 123], axis=0) # 256 is out of range
    return x

m = Model()

input_shape = [256]
x1 = tf.ones(input_shape)

# Call model
y = m(x1)
print(y)
# tf.Tensor([1. 1. 1. 1. 1. 1.], shape=(6,), dtype=float32), with jit_compile=True
# tf.Tensor([1. 1. 1. 1. 0. 1.], shape=(6,), dtype=float32), without jit_compile=True