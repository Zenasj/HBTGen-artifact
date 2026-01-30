import torch.nn as nn

import torch
from tvm import relay
import onnx

model = torch.nn.PReLU()

onnx_model = "test.onnx"

torch.onnx.export(model, (torch.tensor(1.0),), onnx_model, verbose=True)

mod, params = relay.frontend.from_onnx(onnx.load(onnx_model))

mod = relay.transform.InferType()(mod)

import onnxruntime
import torch

model = torch.nn.PReLU()
onnx_model = "test.onnx"
torch.onnx.export(model, (torch.tensor(1.0),), onnx_model, verbose=True)
i_sess = onnxruntime.InferenceSession(onnx_model)
i_sess.run([], {"input": torch.tensor(1.0).numpy(),})

import onnxruntime
import torch

model = torch.nn.PReLU()
onnx_model = "test.onnx"
torch.onnx.export(model, (torch.tensor(1.0),), onnx_model, verbose=True)
i_sess = onnxruntime.InferenceSession(onnx_model)
i_sess.run([], {"input": torch.tensor(1.0).numpy(),})

print(f'{onnxruntime.__version__=}; {torch.__version__=}')