import torch.nn as nn

import torch

example_module = torch.nn.Conv1d(3, 9, 3)
torch.onnx.export(example_module, (torch.ones(5, 3, 11),), "test.onnx", opset_version=16, export_modules_as_functions=True)