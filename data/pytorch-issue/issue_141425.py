import numpy as np

import torch
import torch.nn as nn

torch.manual_seed(0)
device = "cuda"
dtype = torch.float32
eps = 1e-5
layer = nn.LayerNorm(4, eps=eps, dtype=dtype).to(device)
inp: torch.Tensor = torch.randn((2, 3, 4), dtype=dtype, requires_grad=True).to(device)
inp.retain_grad()
out: torch.Tensor = layer(inp)
out.retain_grad()
criteria = lambda x: 0.5 * x.pow(2).sum()
loss: torch.Tensor = criteria(out)
loss.backward()

with torch.no_grad():
    mean_ = inp.mean(dim=-1, keepdim=True)
    shift = inp - mean_
    var = (shift**2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = shift * rstd
nabla_y = out
hand_weight_grad = (nabla_y * x_hat).sum(dim=(0, 1))
hand_bias_grad = nabla_y.sum(dim=(0, 1))
nabla_x_hat = nabla_y * layer.weight
hand_input_grad = rstd * (
    nabla_x_hat
    - nabla_x_hat.mean(dim=-1, keepdim=True)
    - x_hat * (nabla_x_hat * x_hat).mean(dim=-1, keepdim=True)
)
print("weight.grad check: ", torch.allclose(hand_weight_grad, layer.weight.grad))
print("bias.grad check: ", torch.allclose(hand_bias_grad, layer.bias.grad))
print("input.grad check: ", torch.allclose(hand_input_grad, inp.grad))
# print("input.grad check: ", torch.allclose(hand_input_grad, inp.grad, atol=1e-5))  # float32

import torch
import torch.nn as nn

# Set the device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-5

def relative_error(x, y, eps=1e-12):
    return torch.abs(x - y)/(torch.abs(y)+eps)

def absolute_error(x, y):
    return torch.abs(x - y)

def max_relative_error(x, y, eps=1e-12):
    return relative_error(x, y, eps).max()

def mean_relative_error(x, y, eps=1e-12):
    return relative_error(x, y, eps).mean()

def median_relative_error(x, y, eps=1e-12):
    return relative_error(x, y, eps).median()

def max_absolute_error(x, y):
    return absolute_error(x, y).max()

def mean_absolute_error(x, y):
    return absolute_error(x, y).mean()

def median_absolute_error(x, y):
    return absolute_error(x, y).median()

###############################################################################
# Utility to run forward/backward in double precision to get reference
###############################################################################
def run_double_reference(eps=1e-5):
    torch.manual_seed(0)
    layer_double = nn.LayerNorm(4, eps=eps, dtype=torch.float64).to(device)
    inp_double = torch.randn((2, 3, 4), dtype=torch.float64, requires_grad=True, device=device)

    # Forward and backward
    out_double = layer_double(inp_double)
    loss_double = 0.5 * out_double.pow(2).sum()
    loss_double.backward()

    ref = {
        'weight_grad': layer_double.weight.grad.detach().clone(),
        'bias_grad': layer_double.bias.grad.detach().clone(),
        'input_grad': inp_double.grad.detach().clone(),
    }

    return ref

###############################################################################
# Run reference in double precision
###############################################################################
ref = run_double_reference(eps=eps)

###############################################################################
# Run float32 autograd
###############################################################################
torch.manual_seed(0)
layer_float32 = nn.LayerNorm(4, eps=eps, dtype=torch.float32).to(device)
inp_float32 = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True, device=device)

out_float32 = layer_float32(inp_float32)
loss_float32 = 0.5 * out_float32.pow(2).sum()
loss_float32.backward()

auto_weight_grad = layer_float32.weight.grad.detach().clone().to(torch.float64)
auto_bias_grad = layer_float32.bias.grad.detach().clone().to(torch.float64)
auto_input_grad = inp_float32.grad.detach().clone().to(torch.float64)

###############################################################################
# Run manual float32 computations
###############################################################################
torch.manual_seed(0)
layer_manual = nn.LayerNorm(4, eps=eps, dtype=torch.float32).to(device)
inp_manual = torch.randn((2, 3, 4), dtype=torch.float32, requires_grad=True, device=device)
out_manual = layer_manual(inp_manual)
loss_manual = 0.5 * out_manual.pow(2).sum()
loss_manual.backward()

# Manual gradient computation
with torch.no_grad():
    mean_ = inp_manual.mean(dim=-1, keepdim=True)
    shift = inp_manual - mean_
    var = (shift**2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = shift * rstd

nabla_y = out_manual
hand_weight_grad_float32 = (nabla_y * x_hat).sum(dim=(0, 1))
hand_bias_grad_float32 = nabla_y.sum(dim=(0, 1))
nabla_x_hat = nabla_y * layer_manual.weight
hand_input_grad_float32 = rstd * (
    nabla_x_hat
    - nabla_x_hat.mean(dim=-1, keepdim=True)
    - x_hat * (nabla_x_hat * x_hat).mean(dim=-1, keepdim=True)
)

hand_weight_grad = hand_weight_grad_float32.to(torch.float64)
hand_bias_grad = hand_bias_grad_float32.to(torch.float64)
hand_input_grad = hand_input_grad_float32.to(torch.float64)

###############################################################################
# Compute all errors
###############################################################################
def compute_errors(x, y):
    aerr = absolute_error(x, y)
    rerr = relative_error(x, y)
    return {
        'mean_abs_err':   aerr.mean().item(),
        'median_abs_err': aerr.median().item(),
        'max_abs_err':    aerr.max().item(),
        'mean_rel_err':   rerr.mean().item(),
        'median_rel_err': rerr.median().item(),
        'max_rel_err':    rerr.max().item(),
    }

auto_weight_err = compute_errors(auto_weight_grad, ref['weight_grad'])
auto_bias_err   = compute_errors(auto_bias_grad,  ref['bias_grad'])
auto_input_err  = compute_errors(auto_input_grad, ref['input_grad'])

manual_weight_err = compute_errors(hand_weight_grad, ref['weight_grad'])
manual_bias_err   = compute_errors(hand_bias_grad,   ref['bias_grad'])
manual_input_err  = compute_errors(hand_input_grad,  ref['input_grad'])

###############################################################################
# Print in two-column format
###############################################################################
def print_comparison(grad_type, auto_err, manual_err):
    # Columns: Auto on the left, Manual on the right
    print(f"{grad_type} Grad:")
    print(f"  mean abs err:   {auto_err['mean_abs_err']:<15} {manual_err['mean_abs_err']}")
    print(f"  median abs err: {auto_err['median_abs_err']:<15} {manual_err['median_abs_err']}")
    print(f"  max abs err:    {auto_err['max_abs_err']:<15} {manual_err['max_abs_err']}")
    print(f"  mean rel err:   {auto_err['mean_rel_err']:<15} {manual_err['mean_rel_err']}")
    print(f"  median rel err: {auto_err['median_rel_err']:<15} {manual_err['median_rel_err']}")
    print(f"  max rel err:    {auto_err['max_rel_err']:<15} {manual_err['max_rel_err']}")
    print()

print("Final Gradient Differences vs Float64 Reference:\n")
print("                         Auto (float32)        Manual (float32)\n")
print_comparison("Weight", auto_weight_err, manual_weight_err)
print_comparison("Bias",   auto_bias_err,   manual_bias_err)
print_comparison("Input",  auto_input_err,  manual_input_err)