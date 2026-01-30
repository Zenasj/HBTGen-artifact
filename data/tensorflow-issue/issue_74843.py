from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def check_dynamic_shapes(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input Details:")
    for detail in input_details:
        print(f"Name: {detail['name']}, Shape: {detail['shape']}, Dynamic: {any(dim is None for dim in detail['shape'])}")

    print("Output Details:")
    for detail in output_details:
        print(f"Name: {detail['name']}, Shape: {detail['shape']}, Dynamic: {any(dim is None for dim in detail['shape'])}")

# Provide the path to your TFLite model
check_dynamic_shapes('model.tflite')

# Assuming you're using TensorFlow
input_layer = tf.keras.layers.Input(shape=(1, None, None, 3))
# Rest of your model layers
model = tf.keras.Model(inputs=input_layer, outputs=outputs)