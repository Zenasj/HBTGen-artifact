from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

inputs = tf.keras.Input(shape=(256, 256, 3), name='model_input')
outputs = tf.keras.layers.Conv2D(filters=32, kernel_size=3)(inputs)
model = tf.keras.Model(inputs, outputs)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
open("test_model_tf2.0_by_from_keras_model.tflite", "wb+").write(tflite_model)

import tqdm

def tensorflow_lite():
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate

    import numpy as np

    interpreter = Interpreter(
            'test_model_tf2.0_by_from_keras_model.tflite',
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')], #with or without it
        )
    interpreter.allocate_tensors()
   
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    for _ in tqdm.tqdm(range(100000)):
        image = np.zeros((1, 256, 256, 3,), dtype=np.float32)
        set_input_tensor(interpreter, image)
        interpreter.invoke()
        output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

def main():
    tensorflow_lite()
if __name__ == '__main__':
    main()