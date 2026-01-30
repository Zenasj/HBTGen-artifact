import torch
import torch.nn as nn

cuda0 = torch.device('cuda:0')
x = torch.tensor([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]]).to(device=cuda0)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 1, 2)
        ).to(device=cuda0)

    def forward(self, cond, **kwargs):
        if cond:
            return self.net(kwargs['input'])
        else:
            return torch.tensor(0).to(device=cuda0)


module = MyModule()

# Monkey-patch forward
tmp = module.forward
module.forward = lambda cond, input: tmp(cond, input=input)

torch.onnx.export(module,
                  args=(torch.tensor(True).to(device=cuda0), {'input': x}),
                  f='sample.onnx', input_names=['input'], output_names=['output'], export_params=True)