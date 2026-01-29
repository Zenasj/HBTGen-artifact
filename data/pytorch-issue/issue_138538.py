# torch.rand(64, 128, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, ioc):
        super().__init__()
        self.fc1 = nn.Linear(ioc, ioc, bias=False)
        self.fc2 = nn.Linear(ioc, ioc, bias=False)
        self.fc3 = nn.Linear(ioc, ioc, bias=False)
        self.fc4 = nn.Linear(ioc, ioc, bias=False)

        self.grad_acc_hooks = []
        self.grad_acc = []
        self.params = [self.fc1.weight, self.fc2.weight, self.fc3.weight, self.fc4.weight]
        for i, param in enumerate(self.params):

            def wrapper(param):
                param_tmp = param.expand_as(param)
                grad_acc = param_tmp.grad_fn.next_functions[0][0]

                def grad_acc_hook(*notneeded):
                    # Placeholder for gradient synchronization
                    pass

                self.grad_acc.append(grad_acc)
                self.grad_acc_hooks.append(grad_acc.register_hook(grad_acc_hook))

            wrapper(param)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x.sum()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(128)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    bs = 64
    ioc = 128
    return torch.randn([bs, ioc], dtype=torch.float32)

