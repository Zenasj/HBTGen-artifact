import tensorflow as tf
print(tf.__version__)
from keras import layers

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
        self.const = tf.constant(True, shape=[2,2], dtype=tf.bool)

    @tf.function
    def __call__(self, x):
        x = tf.logical_or(self.const, x) # works fine
        x = tf.reshape(x, [2,2,1,1]) # after reshape the result is wrong
        return x


inp = {
    "x": tf.constant(True, shape=[2], dtype=tf.bool),
}
m = MyModule()

out = m(**inp)
print(f'{out}')

runner = get_tflite_callable(m, inp) # Error!
out = runner(**inp)
print(f'{out}')