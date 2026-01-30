from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
train_data, test_data = mnist.load_data()

pre_process = lambda x: x / 255.0
num_calib = 1000
calib_data = pre_process(
            train_data[0][0 : num_calib].astype(np.float32)
        )

model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28)),
                tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=12, kernel_size=(3, 3), activation=tf.nn.relu
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
model.summary()

train_images = pre_process(train_data[0])
train_labels = train_data[1]
test_images = pre_process(test_data[0])
test_labels = test_data[1]
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

def _get_calib_data_func():
  def representative_data_gen():
    for input_value in calib_data:
      input_value = np.expand_dims(input_value, axis=0).astype(np.float32)
      yield [input_value]

  return representative_data_gen

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.representative_dataset = _get_calib_data_func()

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model_INT8 = converter.convert()

input = tf.keras.layers.Input(shape=(225, 225, 3), batch_size=1)