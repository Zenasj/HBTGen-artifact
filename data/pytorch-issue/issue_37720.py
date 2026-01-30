import torch
import torch.nn as nn

class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()
        self.num_batches_tracked = 0
    def forward(self, x):
        self.num_batches_tracked = 1
        return x
model = MyClass()
img = torch.zeros((2, 3,))
torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img) # OK
model = torch.jit.script(model)
torch.onnx.export(model, img, 'f.onnx', verbose=False, opset_version=11, example_outputs=img) # fail