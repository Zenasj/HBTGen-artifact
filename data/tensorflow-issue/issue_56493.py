import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

BATCH_SIZE = 1
NUM_SAMPLES = 3
SAMPLE_SIZE = 4
LSTM_UNITS = 2


def convert_model(model: tf.keras.Model, representative_dataset_gen) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def dataset_example(num_samples: int = 100):
    for _ in range(num_samples):
        yield [
            tf.random.uniform(shape=(1, NUM_SAMPLES, SAMPLE_SIZE), minval=-1, maxval=1)
        ]


def get_model() -> tf.keras.Model:
    input_layer = tf.keras.layers.Input(
        shape=(NUM_SAMPLES, SAMPLE_SIZE), batch_size=BATCH_SIZE
    )
    x = tf.keras.layers.LSTM(
        units=LSTM_UNITS,
        return_sequences=True,
    )(input_layer)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(input_layer, x, name="LSTM")


def test_bug_reproducer() -> None:
    # Get Keras model
    lstm_model = get_model()
    lstm_model.summary()

    # Convert model to INT8 TFLite
    tflite_lstm_model = convert_model(lstm_model, dataset_example)

    # Test Keras model
    test_data = np.random.uniform(-1, 1, (BATCH_SIZE, NUM_SAMPLES, SAMPLE_SIZE))
    print("> Testing Keras LSTM with input data:")
    print(test_data)
    result = lstm_model.predict(test_data)
    print("> got Keras result:")
    print(result)

    # Test INT8 TFLite model
    interpreter = tf.lite.Interpreter(model_content=tflite_lstm_model)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()[0]
    input_details = interpreter.get_input_details()[0]
    input_scale = input_details["quantization_parameters"]["scales"][0]
    input_zp = input_details["quantization_parameters"]["zero_points"][0]
    output_scale = output_details["quantization_parameters"]["scales"][0]
    output_zp = output_details["quantization_parameters"]["zero_points"][0]
    test_data_int8 = np.round((test_data / input_scale) + input_zp).astype(np.int8)
    print("> Testing TFLite INT8 LSTM with input data:")
    print(test_data_int8)
    interpreter.set_tensor(input_details["index"], test_data_int8)
    interpreter.invoke()
    result_int8 = interpreter.get_tensor(output_details["index"])
    print("> got TFlite INT8 result:")
    print(result_int8)
    result_float = (result_int8 - output_zp) * output_scale
    print("> got TFlite float result:")
    print(result_float)


test_bug_reproducer()