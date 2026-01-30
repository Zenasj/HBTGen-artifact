import torch
import numpy as np
import random

torch.onnx.export(model.cpu(), X.cpu().view(1536,1), "model.onnx", verbose=False, input_names=['model_inputs'], output_names=['model_outputs'], dynamic_axes={'model_inputs':[0], 'model_outputs':[0]}) # note dummy input has len 1536

# test seq len is only 1024
seq_len_test = 1024
data = np.random.randint(0,1000,1024)
x_test = np.array(data).reshape(-1,1)

sess = rt.InferenceSession(r"model.onnx", providers=['CPUExecutionProvider'])
output = sess.run(None, {'model_inputs': x_test})[0]