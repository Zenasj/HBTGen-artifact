import numpy as np
import random
import tensorflow as tf

class TestModel2(object):
    def __init__(self):
        self.sess = tf.Session()

        self.input_placeholder = tf.placeholder(tf.float32, [4, 2])

        self.data = tf.constant([[0, 1, 2, 3],
                                 [4, 5, 6, 7],
                                 [8, 9, 10, 11],
                                 [12, 13, 14, 15]], tf.float32, name='data')

        self.data_slice = self.data[:, :2]

        # self.output_node = self.input_placeholder * self.data_slice

        self.sess.run(tf.global_variables_initializer())

    def test_convert_tflite(self):
        print('loading from session')
        converter = tf.contrib.lite.TocoConverter.from_session(self.sess, [self.input_placeholder],
                                                               [self.data_slice])
        print('converting to tflite')
        tflite_model = converter.convert()
        open('dummy.tflite', 'wb').write(tflite_model)

    def test_run(self, inputs):
        output = self.sess.run([self.data_slice], feed_dict={self.input_placeholder: inputs})
        print(output)
        print('-')

class TestTFLite(object):
    def __init__(self, path):
        # Load TFLite model and allocate tensors.
        interpreter = tf.contrib.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print('input tensor:', input_details[0])
        print('output tensor:', output_details[0])

        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details

    def test_run(self, inputs):
        self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output)
        # output = self.interpreter.get_tensor(self.output_details[1]['index'])
        # print(output)

if __name__ == '__main__':
    np.random.seed(1000)
    inputs = np.random.randn(4, 2).astype(np.float32)
    tr = TestModel2()
    tr.test_run(inputs)
    tr.test_convert_tflite()

    tl = TestTFLite('dummy.tflite')
    tl.test_run(inputs)

array([[ 0.,  1.],
       [ 4.,  5.],
       [ 8.,  9.],
       [12., 13.]], dtype=float32)