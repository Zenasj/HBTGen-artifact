import torch.nn as nn
import random

import numpy as np
import torch
np.random.seed(0)
lstm = torch.nn.LSTM(3, 3, 8)
example_input = np.random.randn(8, 3, 3).astype(np.float32)
example_input = torch.from_numpy(example_input)
h0 = example_input
c0 = example_input
lstm_output, (lstm_hidden_output, lstm_cell_state_output) = lstm(example_input, (h0, c0))
torch.onnx.export(lstm, (example_input, (h0, c0)), "pytorch_lstm.onnx")

import onnxruntime as ort
import numpy as np
np.random.seed(0)
example_input = np.random.randn(8, 3, 3).astype(np.float32)
h0 = example_input
c0 = example_input
ort_session = ort.InferenceSession("pytorch_lstm.onnx")
input1_name = ort_session.get_inputs()[0].name
input2_name = ort_session.get_inputs()[1].name
input3_name = ort_session.get_inputs()[2].name
outputs = ort_session.run(None, {input1_name: example_input, input2_name: h0, input3_name: c0})