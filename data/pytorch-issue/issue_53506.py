import torch.nn as nn

import torch

myModel = torch.nn.Conv2d(2, 2, 3)

exampleInput = torch.rand(1, 2, 32, 32)
exampleOutput = myModel(exampleInput)

myTorchScriptModel = torch.jit.trace(myModel, exampleInput)

torch.onnx.export(myTorchScriptModel, exampleInput, "myModel.onnx", example_outputs=exampleOutput)