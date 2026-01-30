import math

import tensorflow as tf
print(tf.__version__)


def get_tflite_callable(model, inp_dict):
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        funcs=[model.__call__.get_concrete_function(**inp_dict)],
        trackable_obj=model,
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_bytes = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    runner = interpreter.get_signature_runner()
    return runner

class MyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.const = tf.constant(True, shape=[1,1,2,2,1], dtype=tf.bool)

    @tf.function # (jit_compile=True)
    def __call__(self, x):
        x = tf.math.logical_xor(x, self.const)
        x = tf.squeeze(x, axis=0)
        return x

inp = {
    "x": tf.constant(True, shape=[2,2,2], dtype=tf.bool),
}
m = MyModule()
print(m(**inp))
runner = get_tflite_callable(m, inp)
print(runner(**inp))