from tensorflow.keras import layers

import os

import tensorflow as tf

from pprint import pprint
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Dense


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_input_details_for_model(input_name_1, input_name_2):
    print()
    print("************")
    print("Input names: ", input_name_1, input_name_2)
    x = Input(shape=(7), name=input_name_1)
    y = Input(shape=(17), name=input_name_2)
    x_with_y = Concatenate()([x, y])
    z = Dense(27, name="output")(x_with_y)

    model = Model([x, y], z)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert() 
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    pprint(interpreter.get_input_details())


if __name__ == "__main__":
    tf.Variable(1.0)
    # The order of tensors is preserved.
    print_input_details_for_model("input_word_ids", "input_bboxes")
    # The order of tensors is changed.
    print_input_details_for_model("input_ids", "bboxes")