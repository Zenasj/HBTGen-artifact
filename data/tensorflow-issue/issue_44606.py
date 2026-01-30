import numpy as np
import math
import tensorflow as tf

class CFModel(tf.Module):
  @tf.function(
      input_signature=(
          tf.TensorSpec(shape=[5, 1], dtype=tf.float32, name='i'),
          tf.TensorSpec(shape=[5, 1], dtype=tf.float32, name='a'),
      ))
  def cf_model(self, i, a):
    z = np.zeros(1)
    k = tf.constant(z,dtype = tf.int32,shape=(1,))
    c = lambda i,a,k: tf.math.less(k,10)
    b = lambda i,a,k: (tf.add(i,a),tf.add(a,a),(k+1),)
    r = tf.while_loop(c,b,[i,a,k])
    return r[0]


def main(argv):
  to_save = tf.saved_model.save(
      CFModel(), '/path/to/folder')
  converter = lite.TFLiteConverterV2.from_saved_model(
      '/path/to/folder')
  tflite_model = converter.convert()
  open('/path/to/folder/model.tflite',
       'wb').write(tflite_model)

from_session

from_saved_model

from_session

tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc

from_session

from_saved_model

while_loop

cond_subgraph

netron