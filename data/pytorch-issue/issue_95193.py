import torch.nn as nn

import torch
class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return torch.logit(tensor, eps=0.3)

if __name__ == "__main__":
    tensor = torch.randn([2], dtype=torch.float32, requires_grad=True)
    model = BasicModule()
    model = torch.compile(model)
    fwd_res = model(tensor)
    bwd_tensor = torch.randn([2], dtype=torch.float32)
    fwd_res.backward(bwd_tensor)
    print(fwd_res)
    print(tensor.grad)