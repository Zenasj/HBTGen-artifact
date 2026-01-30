import torch
import torch.nn as nn

def forward(weight, inp):
    weight.storage().resize_(full_size)
    weight.copy_(...)
    out = aten.mm(weight, inp)
    # first argument is the user fw output, second argument is the saved activation
    weight.storage().resize_(0)
    return out, weight.t()

def forward_functionalized(weight, inp):
    weight_updated = functional_resize(weight, full_size)
    weight_updated2 = weight_updated.copy(...)
    out = aten.mm(weight_updated2, inp)
    weight_updated3 = functional_resize(weight_updated2, 0)
    # input mutation: put a mutable resize in the graph
    weight.storage().resize_(0)
    # first argument is the user fw output, second argument is the saved activation
    return out, weight_updated3.t()

def backward_graph_after_compiled_autograd_runs(tangent, saved_activation, closed_over_bw_hook_tensor):
    closed_over_bw_hook_tensor.storage().resize_(full_size)
    closed_over_bw_hook_tensor.copy_(...)  # copy in all gathered data
    grad_inp = aten.mm(tangent, saved_activation)

def forward(self, L_inputs_0_ : torch.Tensor, L_inputs_1_ : torch.Tensor, L_inputs_2_ : torch.nn.parameter.Parameter):
   ...