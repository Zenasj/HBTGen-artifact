import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 20)
    
    def forward(self, x, trigger=False, y=None):
        x = self.lin(x)
        
        if trigger and y is not None:
            x = x + y
        
        return x

model = MyModule()

x = torch.rand(4, 10)
y = torch.rand(4, 20)

res = model(x, y=y)

inp = {"x": x, "y": y}

traced_model = torch.jit.trace(model, example_kwarg_inputs=inp)

res = traced_model(**inp)

# does work
torch.onnx.export(
    model,
    (inp,),
    "model.onnx"
)

# does not work
torch.onnx.export(
    traced_model,
    (inp,),
    "model.onnx"
)