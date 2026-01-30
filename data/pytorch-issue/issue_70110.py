import torch.nn as nn

import torch
import torch.onnx
import onnx

module = torch.nn.Embedding(20, 40)
inputs = torch.zeros(2, dtype=torch.int)
torch.onnx.export(module, (inputs, ), "model.onnx", input_names=["ids"],
                  dynamic_axes={"ids": {0: "batch"}}
                 )
onxx_model = onnx.load("model.onnx")
print(onxx_model.graph.output)