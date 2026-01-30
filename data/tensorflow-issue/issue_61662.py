import random
import tensorflow as tf

class CppTfTest(tf.Module):

    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def call(self):

        frames = tf.range(600)

        bpm = tf.random.uniform(
            tf.TensorShape([600,]),
            minval=0,
            maxval=90,
            dtype=tf.dtypes.float64
            )

        return bpm, frames

cpp_tf_test = CppTfTest()
tf.saved_model.save(
    cpp_tf_test,
    'cpp_tf_test',
    signatures=cpp_tf_test.call.get_concrete_function()
    )

converter = tf.lite.TFLiteConverter.from_saved_model('cpp_tf_test')

converter.target_spec = tf.lite.TargetSpec(
    supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS],
    experimental_select_user_tf_ops=[
        'RandomUniform', 'Mul'
        ]
)

tflite_model = converter.convert()

with open('cpp_tf_test.tflite', 'wb') as f:
  f.write(tflite_model)