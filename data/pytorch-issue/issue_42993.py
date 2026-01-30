import onnxruntime
import torch
import torch.nn as nn

class dummyNet(nn.Module):
    def __init__(self):
        super(dummyNet, self).__init__()
        self.n =  nn.Conv2d(1, 35, 1)

    def forward(self, x):
        x = self.n(x)
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.flatten(3, 4)
        return x

n = dummyNet()
x = torch.randn(1,1,1024,2)
ort_x = torch.randn(1,1,4096,2).numpy()

torch.onnx.export(n,
                             (x,),
                             "d.onnx",
                             opset_version=12,
                             input_names=['in'],
                             output_names=['out'],
                             dynamic_axes={'in': [0, 2], 'out': [0,2]})

ort_session = onnxruntime.InferenceSession("d.onnx")
ort_inputs = {}
ort_inputs[ort_session.get_inputs()[0].name] = ort_x
ort_outs = ort_session.run(None, ort_inputs)