import torch.nn.functional as F

import torch
import torch.nn as nn

def packed_tensor_from_jagged(tensor):
    offsets = tensor.offsets()
    return torch.cat([t for t in tensor.unbind()], dim = 0), offsets

def modulate(x, shift, scale):

    if scale is not None: # comment out this line => no error
        x = x * (1 + scale) # comment out this line => no error
    if shift is not None:
        x = x + shift
    return x

class test_model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.modulation = nn.Linear(dim, 2 * dim)

    def forward(self, x, c): # x is a ragged tensor (batch_size=4, j, dim=64), c is a regular tensor (batch_size=4, dim=64)
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1) # I think it has something to do with this unsqueeze

        return modulate(x, shift, scale)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = test_model(64).to(device)

### This seems to work fine
batch =torch.randn(4, 512, 64, device=device, requires_grad=True) # batch_size=4, j=512, dim=64
c = torch.randn(4, 64, device=device, requires_grad=True) # batch_size=4, dim=64

output = model(batch, c)
loss = output.sum(dim=-1).mean()
loss.backward()
###

### Bug here, when using nested tensors
batch = torch.nested.nested_tensor([torch.randn(64, 64), torch.randn(128, 64), torch.randn(256, 64), torch.randn(512, 64)], device=device, requires_grad=True, layout=torch.jagged) # batch_size=4, j=jagged, dim=64
c = torch.randn(4, 64, device=device, requires_grad=True) # batch_size=4, dim=64

output = model(batch, c)
output, offsets = packed_tensor_from_jagged(output)
loss = output.sum(dim=-1).mean()
loss.backward() 
# This last line throws an error (Function AddBackward0 returned an invalid gradient at index 0 - got [1, 4, 64] but expected shape compatible with [4, 1, 64])
###

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v)