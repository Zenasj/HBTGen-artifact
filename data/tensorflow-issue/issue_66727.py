from tensorflow.keras import layers

import tensorflow as tf
import keras

input_shape = [16, 1]
output_shape = [13, 1]

tf_input = keras.Input(input_shape[1:], batch_size=input_shape[0])


class MyMatMul(keras.layers.Layer):
    def call(self, tf_input):
        tf_output = tf.matmul(tf.ones((13, 16)), tf_input)
        return tf_output

tf_output = MyMatMul()(tf_input)

model = keras.Model(inputs=[tf_input], outputs=[tf_output])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)