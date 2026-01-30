import torch
import torch.nn as nn

@torch.no_grad()
def do_gather(param):
  # Simulating the deepspeed gather
  r = torch.ones(param.shape, dtype=param.dtype, device=param.device)
  # reduce_partition_and_remove_grads will not be not called if used
  param.set_(r)
  # reduce_partition_and_remove_grads will be called if used
  # param.data = r.data

class SomeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(5, 5))

    def forward(self, x):
        do_gather(self.param)
        return self.param ** 2 * x


model = SomeModule()

input = torch.randn(5,5)

_grad_accs = []
_grad_acc_hooks = []

def wrapper(param):
    print("HAIFENG wrapper PARAM:", id(param))
    param_tmp = param.expand_as(param)
    grad_acc = param_tmp.grad_fn.next_functions[0][0]

    def reduce_partition_and_remove_grads(*notneeded):
        print("HAIFENG reduce_partition_and_remove_grads PARAM:", id(param))

    _grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
    # needs to record otherwise, will not be called?
    _grad_accs.append(grad_acc)


model.train()                   
model.zero_grad()

wrapper(model.param)

print("FROWARD")
output = model(input)
loss = output.sum()
print(loss)

print("BACKWARD")
loss.backward()
print("param GRAD", model.param.grad)