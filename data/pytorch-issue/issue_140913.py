import random

# compare grads from non-compiled GN and compiled GN
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
torch.random.manual_seed(0)

TEST_BN = False

DTYPE = torch.bfloat16
# this config causes high NHWC triton errors
B = 256 # feel free to try out different values of B
C = 64*256 // B
G = C // 4
H = W = 64

create_norm_layer = lambda: nn.GroupNorm(G, C).cuda()

x1 = torch.rand((B, C, H, W)).cuda().to(memory_format=torch.channels_last).to(DTYPE)
x2 = x1.clone()
x3 = x1.clone().contiguous()
x4 = x1.clone().contiguous()

x1.requires_grad_(True)
x2.requires_grad_(True)
x3.requires_grad_(True)
x4.requires_grad_(True)

norm_layer_1 = create_norm_layer().to(DTYPE) # NHWC torch
norm_layer_1.weight.data.uniform_()
norm_layer_1.bias.data.uniform_()

norm_layer_2 = create_norm_layer().to(DTYPE) # NHWC triton
norm_layer_2.weight.data.copy_(norm_layer_1.weight.data)
norm_layer_2.bias.data.copy_(norm_layer_1.bias.data)
norm_layer_2 = torch.compile(norm_layer_2)

norm_layer_3 = create_norm_layer().to(DTYPE) # NCHW torch
norm_layer_3.weight.data.copy_(norm_layer_1.weight.data)
norm_layer_3.bias.data.copy_(norm_layer_1.bias.data)

norm_layer_4 = create_norm_layer().to(DTYPE) # NCHW triton
norm_layer_4.weight.data.copy_(norm_layer_1.weight.data)
norm_layer_4.bias.data.copy_(norm_layer_1.bias.data)
norm_layer_4 = torch.compile(norm_layer_4)

def err_fn(x, y):
    return (x - y).abs().max().item()

dy = torch.rand((B, C, H, W)).cuda().to(memory_format=torch.channels_last, dtype=DTYPE)
dyc = dy.contiguous()

outs = [
        [norm_layer_1, x1, dy, 'Channels L Ptorch'],
        [norm_layer_2, x2, dy, 'Channels L Triton'],
        [norm_layer_3, x3, dyc, 'Channels F Ptorch'],
        [norm_layer_4, x4, dyc, 'Channels F Triton'],
]
labels = [val[-1] for val in outs]

for i, (layer, inp, dy_, _desc) in enumerate(outs):
    layer.weight.grad = None
    layer.bias.grad = None
    inp.grad = None
    out = layer(inp)
    out.backward(dy_)
    outs[i].insert(2, out)

x_errs = np.zeros((len(outs), len(outs)))
w_grad_errs = np.zeros((len(outs), len(outs)))
b_grad_errs = np.zeros((len(outs), len(outs)))
x_grad_errs = np.zeros((len(outs), len(outs)))
for i, (layer1, act1, out1, dy_, desc1) in enumerate(outs):
    for j, (layer2, act2, out2, dy_, desc2) in enumerate(outs):
        x_errs[i][j] = err_fn(out1, out2)
        w_grad_errs[i][j] = err_fn(layer1.weight.grad, layer2.weight.grad)
        b_grad_errs[i][j] = err_fn(layer1.bias.grad, layer2.bias.grad)
        x_grad_errs[i][j] = err_fn(act1.grad, act2.grad)
        if i == 1 and j == 0: # view CL Triton GN with CL Pytorch GN
            diff = (layer1.weight.grad- layer2.weight.grad).flatten()
            fig2, ax2 = plt.subplots()
            ax2.set_title('Weight Grad Channel Idxs vs. Error (Triton weight grad - Torch weight grad)')
            ax2.scatter(np.arange(C), diff.cpu().float().numpy())
            fig2.show()

fig, ax = plt.subplots(2,2, figsize=(15,5))
fig.suptitle(f'GroupNorm (G={G}), input shape: ({B}, {C}, {H}, {W})')
for sub_ax, err, desc in zip(
        [*ax[0], *ax[1]],
        [x_errs, w_grad_errs, b_grad_errs, x_grad_errs],
        ['Fwd Out Errors', 'W Grad Errors', 'B Grad Errors', 'X Grad Errors']
        ):
    im = sub_ax.imshow(err)
    sub_ax.set_title(desc)
    sub_ax.set_xticks(np.arange(len(labels)))
    sub_ax.set_yticks(np.arange(len(labels)))
    sub_ax.set_xticklabels(labels)
    sub_ax.set_yticklabels(labels)
    plt.colorbar(im, ax=sub_ax)
# Rotate x-axis labels for better readability
for arow in ax:
    for a in arow:
        plt.setp(a.get_xticklabels(), rotation=45, ha="right")
plt.show()