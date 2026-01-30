from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np

import tensorflow as tf
print("TENSORFLOW VERSION:", tf.__version__)
from tensorflow.python.keras.engine.input_layer import Input
import tensorflow_model_optimization as tfmo

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras import Model


def calc_mse(target, pred, dtype="float"):

    if dtype=="float":
        scale = 255 ** 2
    elif dtype=="int":
        scale = 1
    else:
        raise NotImplementedError("dtype must be float or int.")

    mse = tf.reduce_mean(tf.pow(target - pred, 2)) * scale
    return mse


def validate_tflite(interpreter, dataset):
    mse = 0

    for img, img in dataset:

        # Preprocess image
        input_details = interpreter.get_input_details()[0]
        scale, zero_point = input_details['quantization']
        tflite_integer_input = img / scale + zero_point
        tflite_integer_input = tf.cast(tflite_integer_input, input_details['dtype'])
        interpreter.set_tensor(input_details['index'], tflite_integer_input)

        interpreter.invoke()

        output_details = interpreter.get_output_details()[0]
        tflite_integer_output = interpreter.get_tensor(output_details['index'])

        tflite_integer_input = tf.cast(tflite_integer_input, "float32")
        tflite_integer_output = tf.cast(tflite_integer_output, "float32")

        mse += calc_mse(tflite_integer_input, tflite_integer_output, dtype="int")

    return mse / len(dataset)


if __name__ == "__main__":

    tfmodel_old = tf.keras.models.load_model("simpleconv")

    tfmodel = tf.keras.Sequential([
        Input((32, 32, 3)),
        Conv2D(128, 3, padding="same", name="Conv1"),
        ReLU(name="Act1"),
        Conv2D(3, 3, padding="same", name="Conv2")])
    
    tfmodel.set_weights(tfmodel_old.get_weights())


    cifar = tf.keras.datasets.cifar10
    (train_images, _), (test_images, _) = cifar.load_data()
    train_images = train_images[:1000].astype(np.float32) / 255.0
    test_images = test_images[:1000].astype(np.float32) / 255.0

    cifar_train = tf.data.Dataset.from_tensor_slices((train_images, train_images)).map(
        lambda x, y: (tf.expand_dims(x, axis=0), tf.expand_dims(y, axis=0)))

    tfmodel.compile(optimizer="adam", loss=calc_mse)
    mse = tfmodel.evaluate(cifar_train)

    print(f"tf savedmodel mse: {mse}")

    # Finetune tensorflow model with QAT
    quantise_model = tfmo.quantization.keras.quantize_model
    qa_model = quantise_model(tfmodel)
    adam = tf.keras.optimizers.Adam(learning_rate=1e-6)
    qa_model.compile(optimizer=adam, loss=calc_mse)

    # NOTE: check mse here
    qa_model.fit(cifar_train)

    # Convert to TFLite and quantise
    converter = tf.lite.TFLiteConverter.from_keras_model(tfmodel)

    # Quantise to int8 without float fallback
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def create_repr_data_gen(data):
            
        def representative_data_gen():
            for input_arr, target in data:
                yield [input_arr]
        return representative_data_gen

    converter.representative_dataset = create_repr_data_gen(cifar_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    quantised_qa_model = converter.convert()
    print("TFLite conversion done!")

    # Validate TFLite model and check mse again
    interpreter = tf.lite.Interpreter(model_content=quantised_qa_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details, "\n", output_details)
    mse_tflite = validate_tflite(interpreter, cifar_train)
    print(f"TFLite mse: {mse_tflite}")