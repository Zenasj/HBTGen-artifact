import math
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np


def create_not_fully_quantizable_model():
    """Set up the model"""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(5), batch_size=1),
            tf.keras.layers.Dense(5, input_shape=(1, 5)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.Lambda(lambda x: tf.math.floormod(x, 3)),
        ]
    )

    return model


def dataset_generator(shape):
    """Set up the training data"""
    x_train = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [2.0, 0.0, 2.0, 3.0, 4.0],
            [0.0, 3.0, 2.0, 3.0, 4.0],
            [4.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 1.0, 2.0, 3.0, 4.0],
        ],
        dtype="float32",
    )

    def generator():
        """Dataset generator for post-training calibration."""
        for x in x_train:
            yield [np.reshape([x], shape)]

    return generator


def main():
    filename = "test_partial_quant"

    model = create_not_fully_quantizable_model()
    model.save(filename)

    # Post-training conversion.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.experimental_new_quantizer = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator(model.input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    # Save result
    tflite_file = filename + ".tflite"
    open(tflite_file, "wb").write(tflite_quant_model)
    print(f"TFLite file saved to '{tflite_file}'.")


if __name__ == "__main__":
    main()