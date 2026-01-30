from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /
                          tf.cast(x_train.shape[0], dtype=tf.float32))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tfp.layers.DenseFlipout(
        10, kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.softmax
    ),
])

optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer, loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                              tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = tflite_converter.convert()

tflite_interpreter = tf.lite.Interpreter(model_content=tflite_model)
tflite_interpreter.allocate_tensors()