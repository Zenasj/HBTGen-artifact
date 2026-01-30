from tensorflow.keras import layers

import tensorflow as tf
import keras

input0_shape = [1, 5]
input1_shape = [1, 5, 7]
output_shape = [1, 1, 7]

tf_input0 = keras.Input(input0_shape[1:], batch_size=1)
tf_input1 = keras.Input(input1_shape[1:], batch_size=1)


class MyMatMul(keras.layers.Layer):
    def call(self, tf_input0, tf_input1):
        # -> [1, 1, 5]
        tf_input0_rank3 = tf.expand_dims(tf_input0, [1])

        # [1, 1, 5] x [1, 5, 7] -> [1, 1, 7]
        tf_output_rank3 = tf.linalg.matmul(tf_input0_rank3, tf_input1)

        # -> [1, 7]
        tf_output = tf.squeeze(tf_output_rank3, [1])

        return tf_output

tf_output = MyMatMul()(tf_input0, tf_input1)

model = keras.Model(inputs=[tf_input0, tf_input1], outputs=[tf_output])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

py
converter.experimental_enable_resource_variables = True

py
import torch
import torch.nn as nn
import ai_edge_torch


class MyMatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        x0_rank3 = torch.unsqueeze(x0, 1)
        out = x0_rank3 @ x1
        out = torch.squeeze(out, 1)
        return out


input0_shape = (1, 5)
input1_shape = (1, 5, 7)
model = MyMatMul()
sample_inputs = (torch.randn(*input0_shape), torch.randn(*input1_shape))

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export("my_mat_mul.tflite")