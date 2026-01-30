import torch

def pack_hook(x):
    return quantize(x)

# You can actually return whatever you want from the pack_hook and it will
# be passed to this function.
def unpack_hook(quant_x):
    return dequantize(quant_x)

with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    # Every SavedVariable in this context will use the above hooks
    your_model(some_input)

# 1. out-of-place, saving input
x = 2 * torch.ones(1, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    y = x.pow(2)
# the input of pack is tensor([2])

# 2. inplace, saving input
x = 2 * torch.ones(1, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    x.pow_(2)
# the input of pack is tensor([2])

# 3. out-of-place, saving output
x = 2 * torch.ones(1, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    y = x.exp()
# the input of pack is tensor([7.3891])

# 4. inplace, saving output
x = 2 * torch.ones(1, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    x.exp_()
# the input of pack is tensor([7.3891])