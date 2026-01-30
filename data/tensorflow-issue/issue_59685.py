import tensorflow as tf
from tensorflow.python import pywrap_mlir
from pathlib import Path


class MyModel(tf.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.w = tf.Variable(5.0, name='weight')
    self.b = tf.Variable(0.0, name='bias')

  @tf.function
  def train(self, x):
    self.w.assign_add(x)
    self.b.assign_add(x)
    return self.w


m = MyModel()
m.train(tf.constant(3.0))
m.train(tf.constant(4.0))
tf.saved_model.save(m, '/tmp/simple-model')


def convert_to_hlo(model_path: str):
  result = pywrap_mlir.experimental_convert_saved_model_to_mlir(
      model_path, "", show_debug_info=False)
  pipeline = ["tf-lower-to-mlprogram-and-hlo"]
  result = pywrap_mlir.experimental_run_pass_pipeline(
      result, ",".join(pipeline), show_debug_info=False)
  return result

Path("/tmp/simple-model.mlir").write_text(
  convert_to_hlo("/tmp/simple-model"))