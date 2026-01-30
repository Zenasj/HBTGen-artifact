import torch.nn as nn

import torch


class Net(torch.nn.Module):
    def forward(self, x):
        return torch.nn.PReLU()(x)


net = Net().eval()

x = torch.zeros((), dtype=torch.float32)

with torch.no_grad():
    y_trh = net(x)
    torch.onnx.export(net, x, "output.onnx", input_names=['inp'], output_names=[
                      'out'], verbose=True, opset_version=14)

import onnxruntime as ort
sess = ort.InferenceSession(
    "output.onnx", providers=['CPUExecutionProvider'])
y_ort = sess.run(['out'], {'inp': x.numpy()})[0]
assert y_ort.shape == y_trh.shape, 'shape mismatch, ORT is `{}` but PyTorch is `{}`'.format(
    y_ort.shape, y_trh.shape)