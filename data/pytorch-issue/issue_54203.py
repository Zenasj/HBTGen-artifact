import torch as th
import torch.nn as nn
import torch.onnx

class TestScript(nn.Module):
    def __init__(self):
        super(TestScript, self).__init__()
    def forward(self, x):
        return (x, )

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.test = th.jit.script(TestScript())
    def forward(self, x):
        return self.test(x)

x = th.zeros(1, dtype=th.float32)
m = Test()
torch.onnx.export(m, (x,), 'test.onnx')