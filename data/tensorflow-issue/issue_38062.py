from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

""" Step0: configure test
"""
batch_size = 128
num_epochs = 10
calibration_size = 300
input_shape = (28, 28, 1)
num_classes = 10

""" Step1: prepare dataset
"""
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0],) + input_shape).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0],) + input_shape).astype('float32') / 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def calibration_gen():
    for i in range(calibration_size):
        yield [x_train[i].reshape((1,) + input_shape)]


""" Step2: prepare models
"""
trained_models = []
model_sequential = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, 5,
            padding='same',
            activation='relu',
            use_bias=False,
            input_shape=input_shape,
        ),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        tf.keras.layers.Conv2D(
            64, 5,
            padding='same',
            activation='relu',
            use_bias=False,
        ),
        tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Softmax(),
    ],
    name='MnistSequential',
)
trained_models.append(model_sequential)


""" Step3: train models
"""
for model in trained_models:
    print(f'Train {model.name}...')
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=True,
    )


""" Step4: convert models
"""
converted_models = []
for model in trained_models:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()
    tflite_path = f'./{model.name}.tflite'
    open(tflite_path, "wb").write(tflite_model)
    converted_models.append(tflite_path)

with quantize_scope():
    for model in trained_models:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_gen

        tflite_model = converter.convert()
        tflite_path = f'./{model.name}_quantized.tflite'
        open(tflite_path, "wb").write(tflite_model)
        converted_models.append(tflite_path)

""" Step5: evaluate keras models & tflite models
"""
x_test = x_test[:calibration_size, :]
y_test = y_test[:calibration_size, :]

for model in trained_models:
    total_seen = 0
    num_correct = 0

    for img, label in zip(x_test, y_test):
        inp = img.reshape((1,) + input_shape)
        total_seen += 1
        predictions = model(inp)
        if np.argmax(predictions) == np.argmax(label):
            num_correct += 1

    score = float(num_correct) / float(total_seen)
    print(f'{model.name} accuracy: {score}')

for tflite_path in converted_models:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    total_seen = 0
    num_correct = 0

    for img, label in zip(x_test, y_test):
        inp = img.reshape((1,) + input_shape)
        total_seen += 1
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        if np.argmax(predictions) == np.argmax(label):
            num_correct += 1

    score = float(num_correct) / float(total_seen)
    print(f'{tflite_path} accuracy: {score}')