import torch
import torch.nn as nn

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
print(out)
out.backward(torch.ones_like(out), retain_graph=True)

def hook_out(module, grad_in, grad_out):
    print("backward hook out")

    def hook_in(module, grad_in, grad_out):
        print("backward hook in")

    inp = torch.eye(4, 5, requires_grad=True)
    out = (inp+1).pow(2).t()
    print(out)
    # out.backward(torch.ones_like(out), retain_graph=True)

model_out = nn.Conv2d(3, 3, 1, 1)
input_out = torch.randn([1, 3, 16, 16])
back_hook_out = model_out.register_full_backward_hook(hook_out)
model_out(input_out).sum().backward()