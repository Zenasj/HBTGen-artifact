from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

print("version is {}, git version is {}".format(tf.version.VERSION, tf.version.GIT_VERSION))


from model import Model

class SimpleModel(Model):
    def __init__(self):
        super().__init__()
        self.model_name = "mnist"
        self.train_data = None
        self.test_data = None
        self.calib_data = None
        self.num_calib = 1000
        # (data preprocessing) Normalize the input image so that
        # each pixel value is between 0 to 1.
        self.pre_process = lambda x: x / 255.0

        self._load_data()
        self._set_path()

    def _load_data(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist

        # _data: (images, labels)
        self.train_data, self.test_data = mnist.load_data()
        self.calib_data = self.pre_process(
            self.train_data[0][0 : self.num_calib].astype(np.float32)
        )

    def train(self):
        cell = tf.keras.layers.GRUCell(3)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28), name='input'),
            #tf.keras.layers.LSTM(32),
            tf.keras.layers.RNN(cell, unroll=True),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')
        ])
        model.summary()

        train_images = self.pre_process(self.train_data[0])
        train_labels = self.train_data[1]
        test_images = self.pre_process(self.test_data[0])
        test_labels = self.test_data[1]
        # Train the digit classification model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            train_images,
            train_labels,
            epochs=1,
            validation_data=(test_images, test_labels),
        )
        # dump SavedModel - ANOTHER BUG HERE!
        #model.save(str(self.savedModel_dir))

        return model

    def eval(self, tflite_model_path: str):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        interpreter.allocate_tensors()

        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # (data preprocessing) Normalize the input image so that
        # each pixel value is between 0 to 1.
        test_images = self.pre_process(self.test_data[0])
        test_labels = self.test_data[1]
        # Run predictions on every image in the "test" dataset.
        prediction_digits = []
        for test_image in test_images:
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_digits.append(digit)

        # Compare prediction results with ground truth labels to calculate accuracy.
        accurate_count = 0
        for index, _ in enumerate(prediction_digits):
            if prediction_digits[index] == test_labels[index]:
                accurate_count += 1
        accuracy = accurate_count * 1.0 / len(prediction_digits)

        return accuracy

    def _get_calib_data_func(self):
        def representative_data_gen():
            for input_value in self.calib_data:
                input_value = np.expand_dims(input_value, axis=0).astype(np.float32)
                yield [input_value]

        return representative_data_gen


if __name__ == "__main__":
    temp = SimpleModel()
    model = temp.train()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = temp._get_calib_data_func()

    # save INT8 tflite
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS]
    converter.experimental_new_converter = True
    tflite_model_INT8 = converter.convert()
    open("lstm_unrolled_int8.tflite", "wb").write(tflite_model_INT8)

    print("INT8 model eval results: {:f}".format(temp.eval("lstm_unrolled_int8")))