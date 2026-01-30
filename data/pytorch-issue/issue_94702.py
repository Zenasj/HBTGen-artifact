import torch
import torch.nn as nn

L, E = (1, 2), 3
x = torch.nested.nested_tensor([
    torch.rand(L[0], E), 
    torch.rand(L[1], E)
])

norm = nn.LayerNorm(E)

x = norm(x)
x = torch.nested.to_padded_tensor(x, padding=0.).mean()
loss = 0. - x
loss.backward() # NotImplementedError: Could not run 'aten::native_layer_norm_backward' with arguments from the 'NestedTensorCPU' backend.