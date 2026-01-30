import torch.nn as nn

import onnx
import torch

class Model(torch.nn.Module):
    def forward(self, a, b):
        return a < b


print(torch.onnx.dynamo_export(Model(), torch.tensor(True), torch.tensor(False)).model_proto)