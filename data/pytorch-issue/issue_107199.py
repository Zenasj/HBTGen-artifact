import torch.nn as nn

# reproducer
import torch
import copy
import torch.utils._pytree as pytree

x = torch.randn(2, 3, 4, 4, 4).cuda().requires_grad_()
x2 = x.contiguous(memory_format=torch.channels_last_3d).cuda().detach().requires_grad_()
model = torch.nn.BatchNorm3d(3, 0.001, 0.7, False).cuda()
model2 = copy.deepcopy(model).to(memory_format=torch.preserve_format).cuda()
output = model(x)
grad_out = torch.randn(output.shape).cuda()
grad_out2 = grad_out.clone().contiguous(memory_format=torch.channels_last_3d).cuda()

output2 = model2(x2)

if isinstance(output2, torch.Tensor):
    outputs2 = (output2,)


# === Do backward pass. ===
diff_outputs = tuple(t for t in outputs2)
params = tuple(p for p in model2.parameters())
inputs=(x2,)
diff_inputs = tuple(
    t
    for t in pytree.tree_flatten((inputs, params))[0]
)
grad_outputs = tuple(
    torch.rand_like(t)
    for t in diff_outputs
)

grad_inputs = torch.autograd.grad(
    diff_outputs,
    diff_inputs,
    grad_outputs=grad_outputs,
    allow_unused=True,
)

for gi in grad_inputs:
    print("gi.shape: ", gi.shape, " gi.stride: ", gi.stride())