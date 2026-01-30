import torch
import torch.nn as nn
# import caffe2.python.onnx.backend as backend
import onnx


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
  
  def forward(self, x):
    return 2 * x


device = 'cuda'

model = Model().eval().to(device)

input_data = torch.tensor([[0.0, 0.0]], device=device)
torch.onnx.export(model, input_data, 'model.onnx', verbose=True)

onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# rep = backend.prepare(onnx_model, device)
# onnx_model_out = rep.run(input_data.numpy())

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(in_c, num_filters, kernel_size=kdim,
                      stride=stride, padding=padding, groups=1,
                      bias=False)

    def forward(self, x):
        out = self.layer1(x)
        return out