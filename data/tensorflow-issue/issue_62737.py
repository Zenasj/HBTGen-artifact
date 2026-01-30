from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np  # 1.26.2
import tensorflow as tf  # 2.8.4
from tensorflow import keras  # 2.8.0
from pathlib import Path


class ImplicitA(keras.layers.Layer):
    def __init__(self, mean, std, name, **kwargs):
        super(ImplicitA, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.impa = self.add_weight(
            name=self.name,
            shape=(1, 1, 1, input_shape[-1]),
            initializer=keras.initializers.RandomNormal(
                mean=self.mean, stddev=self.std
            ),
            trainable=True
        )

    def call(self, x):
        return tf.cast(x, self.impa.dtype) + self.impa

    def get_config(self):
        config = super(ImplicitA, self).get_config()
        config.update(
            {'mean': self.mean, 'std': self.std}
        )
        return config


class ImplicitM(keras.layers.Layer):
    def __init__(self, mean, std, name, **kwargs):
        super(ImplicitM, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.impm = self.add_weight(
            name=self.name,
            shape=(1, 1, 1, input_shape[-1]),
            initializer=keras.initializers.RandomNormal(
                mean=self.mean, stddev=self.std
            ),
            trainable=True
        )

    def call(self, x):
        return tf.cast(x, self.impm.dtype) * self.impm

    def get_config(self):
        config = super(ImplicitM, self).get_config()
        config.update(
            {'mean': self.mean, 'std': self.std}
        )
        return config


def model1():
    inputs = keras.Input(shape=(32, 32, 4))
    x = ImplicitA(mean=0.0, std=0.02, name='impa')(inputs)
    x = keras.layers.Conv2D(
        filters=8, kernel_size=1, strides=1, name='conv'
    )(x)
    x = ImplicitM(mean=0.0, std=0.02, name='impm')(x)
    return keras.Model(inputs=inputs, outputs=x)


def model2():
    inputs = keras.Input(shape=(32, 32, 4))
    x = ImplicitA(mean=0.0, std=0.02, name='impa')(inputs)
    x = keras.layers.Conv2D(
        filters=8, kernel_size=1, strides=1, name='conv'
    )(x)
    return keras.Model(inputs=inputs, outputs=x)


def model3():
    inputs = keras.Input(shape=(32, 32, 4))
    x = ImplicitA(mean=0.0, std=0.02, name='impa')(inputs)
    x = ImplicitM(mean=0.0, std=0.02, name='impm')(x)
    x = keras.layers.Conv2D(
        filters=8, kernel_size=1, strides=1, name='conv'
    )(x)
    return keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    model = model1()
    model.summary()
    model_path = str(Path(__file__).parent / "temp_model.h5")
    model.save(model_path)

    # load model
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "ImplicitA": ImplicitA,
            "ImplicitM": ImplicitM
        }
    )

    # convert to tflite
    in_sh = model.input.shape

    def datagen():
        yield [np.ones((1, in_sh[1], in_sh[2], in_sh[3])).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = datagen
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    tflite_model_path = str(Path(__file__).parent / "temp_model.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)