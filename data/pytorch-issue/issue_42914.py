import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, x):
        return x < self.x

def test(*args):
    model = Model(*args)
    dummy_input = [torch.tensor([[10]], dtype=torch.float32)]
    example_outputs = [model(*dummy_input)]
    model = torch.jit.script(model)
    torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)

test(3)
#  File "bug.py", line 16, in test
#    torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)
#  File "/miniconda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 203, in export
#    return utils.export(model, args, f, export_params, verbose, training,
#  File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 86, in export
#    _export(model, args, f, export_params, verbose, training, input_names, output_names,
#  File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 526, in _export
#    graph, params_dict, torch_out = _model_to_graph(model, args, verbose, input_names,
#  File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 350, in _model_to_graph
#    method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
# RuntimeError: Unknown type int encountered in graph lowering. This type is not supported in ONNX export.

test(3.0)
# ...
# RuntimeError: Unknown type float encountered in graph lowering. This type is not supported in ONNX export.

test(torch.tensor(3.0))
# works

test(torch.tensor(3))
# Warning: ONNX Scalar Type Analysis - Scalar types mismatch for tensor inputs of operator onnx::Less. Please report a bug to PyTorch. The scalar type Float of the first tensor is chosen.

import torch

class Model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self, x):
        return x[self.x]

def test(*args):
    model = Model(*args)
    dummy_input = [torch.tensor([[10]], dtype=torch.float32)]
    example_outputs = [model(*dummy_input)]
    model = torch.jit.script(model)
    torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)

test(torch.tensor(0))
# works

test(0)
# RuntimeError: Unknown type int encountered in graph lowering. This type is not supported in ONNX export.

import torch

class Model(torch.nn.Module):
    def __init__(self, x, bias):
        super().__init__()
        self.embeddings = torch.nn.Linear(1, 10, bias=bias)

    def forward(self, x):
        return self.embeddings(x)

def test(*args):
    model = Model(*args)
    dummy_input = [torch.tensor([[3]], dtype=torch.float32)]
    example_outputs = [model(*dummy_input)]
    model = torch.jit.script(model)
    torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)

test(torch.tensor([[3]]), True)
# works

test(torch.tensor([[3]]), False)
# Traceback (most recent call last):
#   File "bug.py", line 20, in <module>
#     test(torch.tensor([[3]]), False)
#   File "bug.py", line 16, in test
#     torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)
#   File "/miniconda/lib/python3.8/site-packages/torch/onnx/__init__.py", line 203, in export
#     return utils.export(model, args, f, export_params, verbose, training,
#   File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 86, in export
#     _export(model, args, f, export_params, verbose, training, input_names, output_names,
#   File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 526, in _export
#     graph, params_dict, torch_out = _model_to_graph(model, args, verbose, input_names,
#   File "/miniconda/lib/python3.8/site-packages/torch/onnx/utils.py", line 350, in _model_to_graph
#     method_graph, params = torch._C._jit_pass_lower_graph(graph, model._c)
# RuntimeError: Unknown type None encountered in graph lowering. This type is not supported in ONNX export.

import torch
model = torch.nn.Linear(1, 10, bias = False)
dummy_input = [torch.tensor([[3]], dtype=torch.float32)]
example_outputs = [model(*dummy_input)]
model = torch.jit.script(model)
torch.onnx.export(model, dummy_input, 'test.onnx', example_outputs=example_outputs)