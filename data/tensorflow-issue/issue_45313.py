import tensorflow as tf
import os


class SimpleModel:
    def infer_tflite(self, features):
        return tf.zeros(shape=(250, 1), dtype=tf.dtypes.int32)


def tf_lite_convert():
    print('TensorFlow version')
    print(tf.__version__)
    model = SimpleModel()

    single_elem = tf.zeros(shape=[1, 20], dtype=tf.dtypes.int64)
    print('Running predictions using tflite inference')
    preds = model.infer_tflite(single_elem)
    print('TFlite inference results')
    print(preds)
    print('Turning to Concrete function')
    new_infer_fn = tf.function(model.infer_tflite, input_signature=[tf.TensorSpec((1, None), dtype=tf.int64)])
    infer_concrete = new_infer_fn.get_concrete_function()

    print('Running inference with concrete function')
    preds = infer_concrete(single_elem)
    print('Concrete Output prediction')
    print(preds)

    print('Saving to Tflite')
    converter = tf.lite.TFLiteConverter.from_concrete_functions([new_infer_fn.get_concrete_function()])

    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_path = os.path.join('', 'lite_model.tflite')

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    tf_lite_convert()