import random
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM

model_name = "1x_LSTM_64_float32"
input_length = 256

class SimpleModel(Model):
  def __init__(self, input_shape, hidden_size):
    super().__init__()
    
    self.lstm = LSTM(hidden_size, return_sequences = True, return_state=True, input_shape = [-1, input_shape] )
    
    self.d1 = Dense(1, input_shape = [-1, hidden_size])

  def call(self, x):
      
    x, h0, c0 = self.lstm(x)
    x = self.d1(x)
    
    return x#, tf.stack([h0, c0])

model = SimpleModel(input_length, 64)

out, states = model(tf.random.uniform([16,43,256]))

print(np.mean(out))

model_path = f"{model_name}.tf"

run_model = tf.function(lambda x: model(x))
BATCH_SIZE = 16
STEPS = 43
INPUT_SIZE = 256
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], tf.float32))

model.save(model_path, save_format = 'tf', signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,
]

tflite_model = converter.convert()
open(f"{model_name}.tflite", "wb").write(tflite_model)

class SimpleModel(Model):
  def __init__(self, input_shape, hidden_size):
    super().__init__()
    
    self.lstm = LSTM(hidden_size, return_sequences = True, return_state=True, input_shape = [-1, input_shape] )
    
    self.d1 = Dense(1, input_shape = [-1, hidden_size])

  def call(self, x):
      
    x, h0, c0 = self.lstm(x)
    x = self.d1(x)
    
    return x, tf.stack([h0, c0])

model = SimpleModel(input_length, 64)

out, states = model(tf.random.uniform([16,43,256]))

py
import torch
from torch import nn
import ai_edge_torch


class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
    
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.d1 = nn.Linear(hidden_size, 1)

    def forward(self, x):
      
        x, (h0, c0) = self.lstm(x)
        x = self.d1(x)
    
        return x, torch.stack([h0, c0])


model = SimpleModel(256, 64)
sample_inputs = (torch.randn(16, 43, 256),)

edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export("simple_model.tflite")