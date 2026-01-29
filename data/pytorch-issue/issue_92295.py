# torch.rand(4, dtype=torch.float32)
import torch
import torch.nn as nn
import functorch
import optree

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        nn.init.ones_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        out = self.fc(x).mean()
        grad = torch.autograd.grad(out, self.parameters(), create_graph=True)
        return grad

class MyModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.base_model = Model(dim)

    def forward(self, x):
        oop_success = False
        try:
            grad_oop = self.base_model(x)
            oop_success = True
        except:
            pass

        func_success = False
        try:
            fmodel, params, buffers = functorch.make_functional_with_buffers(self.base_model)
            out = fmodel(params, buffers, x).mean()
            grad_func = torch.autograd.grad(out, params, create_graph=True)
            func_success = True
        except:
            func_success = False

        fcall_success = False
        try:
            parameters_and_buffers = dict(self.base_model.named_parameters()) | dict(self.base_model.named_buffers())
            parameters_and_buffers = optree.tree_map(
                lambda t: t.clone().detach().requires_grad_(t.requires_grad),
                parameters_and_buffers
            )
            out_fcall = torch.func.functional_call(self.base_model, parameters_and_buffers, x).mean()
            grad_fcall = torch.autograd.grad(
                out_fcall,
                list(parameters_and_buffers.values()),
                create_graph=True
            )
            fcall_success = True
        except:
            fcall_success = False

        return torch.tensor([oop_success, func_success, fcall_success], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.float32)

