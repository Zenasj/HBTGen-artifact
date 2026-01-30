import os

import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize

os.environ['CUDA_VISIBLE_DEVICES'] = ''
MODEL = './test.tflite'


def convert():
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            data = tf.placeholder('float32', [1, 512, 2], name='data')
            indices = tf.placeholder('int64', [1, 1], name='indices')
            output = tf.gather(data, indices, batch_dims=1)

            # Tensor("GatherV2:0", shape=(1, 1, 2), dtype=float32)
            print(output)
            contrib_quantize.experimental_create_eval_graph(
                input_graph=g)
            converter = tf.lite.TFLiteConverter.from_session(
                sess,
                input_tensors=[data, indices],
                output_tensors=[output])
            tflite_model = converter.convert()
            with open(MODEL, "wb") as w:
                w.write(tflite_model)


def load():
    interpreter = tf.lite.Interpreter(MODEL)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()

    # [
    # {'name': 'GatherV2', 'index': 0,
    # 'shape': array([1, 1, 1, 2], dtype=int32),
    # 'dtype': <class 'numpy.float32'>,
    # 'quantization': (0.0, 0)}
    # ]
    print(output_details)


if __name__ == '__main__':
    convert()
    load()