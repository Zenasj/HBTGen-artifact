import tensorflow as tf

class MyModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def my_operation(self, x):
        return tf.cos(x)

model = MyModel()

concrete_func = model.my_operation.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open("nnapi_cos_bug.tflite", "wb") as f:
    f.write(tflite_model)