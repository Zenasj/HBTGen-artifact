import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class a(tf.keras.layers.Layer):
  pass
class b(a):
  pass
class c(b):
  pass
aobj = a()
bobj = b()
cobj = c()
print(callable(aobj))
print(callable(bobj))
print(callable(cobj))