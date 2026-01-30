import torch
import torch.nn as nn
import numpy as np
import random

print(torch.__version__)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedd = nn.EmbeddingBag(500, 12)

    def forward(self, x_user):
        user = self.embedd(x_user)
        return user
a = DummyModel()
batch_size = 10
from torch.onnx import OperatorExportTypes
onnx_torch.export(a, torch.from_numpy(np.random.randint(0, 7, size=(batch_size, 5)).astype(np.int64)),
                  'model.onnx', 
                  input_names=['inp'],
                  output_names=['output'],
                  dynamic_axes={'inp': {0: 'batch_size'},
                                }, verbose=True,
                  operator_export_type=OperatorExportTypes.ONNX,
                  opset_version=10
                 )
onnx_model = onnx.load("model.onnx")
print(onnx.checker.check_model(onnx_model))
a = onnxruntime.InferenceSession('model.onnx')