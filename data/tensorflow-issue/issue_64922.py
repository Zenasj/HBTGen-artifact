from tensorflow import keras

import tensorflow as tf

def build_dummy_model(input_shape, name="dummy_model"):
    inputs = tf.keras.Input(shape=input_shape, name="input_layer")
    x_norm = tf.linalg.normalize(inputs, axis=2, name=f"{name}_norm_pred_bones")[0]
    model = tf.keras.Model(inputs=inputs, outputs=x_norm, name=name)
    return model

input_shape = (26, 3) 
model = build_dummy_model(input_shape)

py
import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=2)


batch_size = 1
input_shape = (batch_size, 26, 3)
model = DummyModel()
sample_inputs = (torch.randn(*input_shape),)

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export("dummy_model.tflite")