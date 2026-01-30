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

class LoopModel2(torch.nn.Module):
    def __init__(self):
        super(LoopModel2, self).__init__()
        self.work = torch.ones(2, 3, dtype=torch.long)
        self.test: bool = True

    def forward(self, x, y):
        if self.test:
            for i in range(int(y)):
                x = x + i
            return x

model = torch.jit.script(LoopModel2())
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', example_outputs=dummy_input)

print("Yeah we made it!")

class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()
        self.num_batches_tracked: int = 0
    def forward(self, x):
        self.num_batches_tracked += 1
        return x

model = MyClass()
img = torch.zeros((2, 3))
torch.onnx.export(model, img, 'f.onnx', example_outputs=img) # OK
model = torch.jit.script(model)
torch.onnx.export(model, img, 'f.onnx', example_outputs=img) # fail

class MyClass(nn.Module):
    def __init__(self):
        super(MyClass, self).__init__()
        self.num_batches_tracked = torch.ones(2, 3, dtype=torch.long)
    def forward(self, x):
        self.num_batches_tracked[[0]] += 1
        return x