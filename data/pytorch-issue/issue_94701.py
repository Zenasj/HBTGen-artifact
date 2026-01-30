import torch
import torch.nn as nn

L, E = (1, 2), 3
x = torch.nested.nested_tensor([
    torch.rand(L[0], E), 
    torch.rand(L[1], E)
])

linear = nn.Linear(E, E)
act = nn.GELU()

x = act(linear(x))
x = torch.nested.to_padded_tensor(x, padding=0.).mean()
loss = 0. - x
loss.backward() # NotImplementedError: Could not run 'aten::gelu_backward' with arguments from the 'NestedTensorCPU' backend.