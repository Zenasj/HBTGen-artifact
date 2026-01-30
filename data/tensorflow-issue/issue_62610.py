from tensorflow.keras import layers

model.tflite

converted_model.tflite

import argparse
import keras
import tensorflow as tf

INPUT_SHAPE = (5, 5, 3)
CONV_FILTERS = 6
CONV_KER_SIZE = (2, 2)
FC_UNITS = 4

def build_model():
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE, batch_size=1),
            keras.layers.Conv2D(CONV_FILTERS, CONV_KER_SIZE),
            keras.layers.Flatten(),
            keras.layers.Dense(FC_UNITS, activation='softmax'),
        ]
    )

    model.summary()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, help="Output filename", default=None)
    parser.add_argument("--totflite", "-t", type=str, help="Output TFlite filename", default=None)
    args = parser.parse_args()
    model = build_model()

    if args.output:
        model.save(args.output)

    if args.totflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(args.totflite, 'wb') as f:
            f.write(tflite_model)