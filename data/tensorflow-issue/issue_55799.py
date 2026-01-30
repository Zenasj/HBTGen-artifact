import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import matplotlib.pyplot as plt


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dilation):
        super(MyLayer, self).__init__()

        self.c0 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         padding='same',
                                         activation=None)

        self.c1 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         dilation_rate=(dilation, 1),
                                         padding='same',
                                         activation=None)

        self.c2 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         padding='same',
                                         activation=None)

    def call(self, x):
        #  input: (B,  time, slices, channels)
        x = self.c0(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


def main():
    dilation = 1    # yields almost equivalent results for filters == 32
    # dilation = 2  # yields different results between tflite and tensorflow for filters == 32
    # filters = 16  # with filters == 16, equivalent results are obtained, for both versions of dilation
    filters = 32  

    # Create a model with a Layer containing three Conv2D Layers
    input_shape = (128, 1, 1)
    x = tf.keras.Input(shape=input_shape, name='input', dtype=tf.float32)

    y = MyLayer(filters=filters, dilation=dilation)(x)

    model = tf.keras.Model(inputs=[x], outputs=[y])

    # Save tensorflow model
    test_tensorflow_model_path = 'test_tensorflow_model/'
    test_tflite_model_file = test_tensorflow_model_path + 'test_tflitemodel.tflite'
    model.save(test_tensorflow_model_path)

    # Load model from disk
    model = tf.keras.models.load_model(test_tensorflow_model_path)

    # create random test data and call model
    input_data = tf.random.normal((1, ) + input_shape, dtype=tf.float32)
    output_tf = model(input_data)

    # Create a TFlite model
    converter = tf.lite.TFLiteConverter.from_saved_model(test_tensorflow_model_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(test_tflite_model_file, 'wb') as f:
        f.write(tflite_model)
    print("TF lite export done")

    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter(test_tflite_model_file)
    interpreter.allocate_tensors()

    # Call TFlite model with same input data as TF-model
    input_details = interpreter.get_input_details()[0]

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    out_tensor_index = interpreter.get_output_details()[0]['index']
    output_tflite = interpreter.get_tensor(out_tensor_index)

    # Compare results
    print('tflite_output: ', output_tflite[0, :, :, 0])
    print('tf_output: ', output_tf[0, :, :, 0])

    print('relative error: ', tf.math.sqrt(2*tf.math.reduce_variance(output_tf-output_tflite) / tf.math.reduce_variance(output_tf+output_tflite)), '%')

    plt.plot(output_tflite[0, :, :, 0])
    plt.plot(output_tf[0, :, :, 0])
    plt.show()


if __name__ == '__main__':
    main()