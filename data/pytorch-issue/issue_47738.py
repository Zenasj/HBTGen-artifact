import torch.nn as nn

import onnxruntime
import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.p = torch.nn.Parameter(torch.zeros([5]))

    def forward(self, x):
        return x + self.p.size()[0]

def main():
    m = MyModule()
    x = torch.tensor(4)
    assert m(x).size() == ()

    torch.onnx.export(m, x, 'test.onnx', opset_version=11,
        input_names=['x'], output_names=['y'])

    sess = onnxruntime.InferenceSession('test.onnx')
    y, = sess.run(['y'], {'x': x.numpy()})
    assert y.shape == (), f"y.shape is {y.shape}"

main()