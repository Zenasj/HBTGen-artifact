import torch.nn as nn

import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d1):
        # To run without warnings, add: d1 = d1.long()
        return torch.full([d1], 1)


dummy_model = DummyModel()
d1 = torch.tensor(2, dtype=torch.int32)
torch.onnx.export(
    dummy_model,
    (d1,),
    f="my_filename.onnx",
    verbose=True,
    input_names=["d1"],
    output_names=["casted_data"],
)

import onnxruntime as ort
import torch
import sys


onnx_model = sys.argv[1]
dummy_data = torch.randint(low=1, high=10, size=[], dtype=torch.int32)

session = ort.InferenceSession(onnx_model)
outputs_onnx = torch.FloatTensor(session.run(None, {"data": dummy_data.numpy()})[0])