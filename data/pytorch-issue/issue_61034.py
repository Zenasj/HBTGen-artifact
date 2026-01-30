import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        

    def forward(self,  index,  A, B):
        A = A.index_put((index,), B, accumulate=True)
        return A

trained_model = Net()
A= torch.ones(5, 3)
B = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1],[1, 1, 1],[1,1,1]], dtype=torch.float)
index = torch.tensor([0, 4, 2,2,4])
input = (index,A,B)
torch.onnx.export(
                trained_model,
                input,
                "net_demo.onnx",
                # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                opset_version=11)

ort_session = onnxruntime.InferenceSession("net_demo.onnx")
ort_inputs = {ort_session.get_inputs()[0].name:index.cpu().numpy(),
    ort_session.get_inputs()[1].name:A.cpu().numpy(),
    ort_session.get_inputs()[2].name: B.cpu().numpy()}

ort_outs = ort_session.run(None, ort_inputs)
print('ort_outs:')
print(ort_outs)