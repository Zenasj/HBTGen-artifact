import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow_graphics.math.optimizer import levenberg_marquardt


def convert_to_tflite(
    model_path: str,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]

    tflite_model = converter.convert()
    assert isinstance(tflite_model, bytes)

    return tflite_model


def tflite_inference(
    inputs: list[tf.Tensor],
    tflite_model: bytes,
) -> list:
    # create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # set inputs
    input_details = interpreter.get_input_details()
    for input_tensor, input_placeholder in zip(inputs, input_details):
        interpreter.set_tensor(input_placeholder["index"], input_tensor)

    # invoke interpreter
    interpreter.invoke()

    # return outputs
    output_details = interpreter.get_output_details()
    return [interpreter.get_tensor(output["index"]) for output in output_details]


def create_lmo_model():
    def f1(x, y):
        return x + y

    def f2(x, y):
        return x * y

    class LMO(tf.keras.layers.Layer):
        def call(self, inputs):
            _, (r1, r2) = levenberg_marquardt.minimize(
                residuals=(f1, f2),
                variables=inputs,
                max_iterations=10,
            )
            return [r1, r2]

    input_x = tf.keras.Input(
        shape=(1, 2),
    )
    input_y = tf.keras.Input(
        shape=(3, 1),
    )
    output = LMO()([input_x, input_y])

    return tf.keras.Model(
        inputs=[input_x, input_y],
        outputs=output,
    )


def main():
    tf.random.set_seed(5)
    x = tf.random.uniform((1, 1, 2))
    y = tf.random.uniform((1, 3, 1))
    inputs = [x, y]

    model = create_lmo_model()
    print(model(inputs))

    tf.saved_model.save(
        model,
        "lmo_model",
    )

    tflite_model = convert_to_tflite("lmo_model")
    print(
        tflite_inference(
            tflite_model=tflite_model,
            inputs=inputs,
        )
    )


if __name__ == "__main__":
    main()