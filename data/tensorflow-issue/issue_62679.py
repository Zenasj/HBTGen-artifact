import tensorflow as tf

tf.reverse(tf.ones((1,), dtype=tf.float32), [])  # no problem

class Foo(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def reverse(self, x):
        #    works fine if axis = [0]
        #    crashes if axis = []
        return tf.reverse(x, axis=[])

foo = Foo()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    funcs=[foo.reverse.get_concrete_function()],
    trackable_obj=foo,
)

tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
# crash
interpreter.get_signature_runner()(x=tf.ones((1,), dtype=tf.float32))