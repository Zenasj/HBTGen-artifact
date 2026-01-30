import torch
import torch.nn as nn

param.storage().resize_(all_gather_size)
param.copy_(all_gather_buffer)
out = aten.matmul(param, param)

out = aten.matmul(all_gather_buffer, all_gather_buffer)

def forward(self, primals_1, primals_2):
    _foreach_copy = torch.ops.aten._foreach_copy.default([primals_1], [primals_2]);  primals_2 = None
    getitem = _foreach_copy[0];  _foreach_copy = None
    mm = torch.ops.aten.mm.default(getitem, getitem);  getitem = None
    t_1 = torch.ops.aten.t.default(primals_1);  primals_1 = None
    return [mm, t_1]

def forward(l_fn_weight, synthetic_local_tmp_0_, inp):
    cat = torch.cat([l_fn_weight, l_fn_weight], 0)
    tracable_create_parameter = torch__dynamo_create_parameter_op_tracable_create_parameter(cat, synthetic_local_tmp_0_)
    result = torch._C._nn.linear(inp, tracable_create_parameter, None)

def forward(zero_sized_param):
    inductor_resize_storage_(zero_sized_param, all_gather_size)
    zero_sized_param.copy_(all_gather_buffer)
    torch._C._nn.linear(input, zero_sized_param.T)
    ...

def f(dummy_param, param_shard):
    allgather = all_gather(param_shard)
    dummy_param.set_(allgather)
    dummy_param_view = dummy_param.view(dummy_param.shape)
    ...
    dummy_param_view.storage_resize_(0)

y = torch.nn.Parameter(x)  # create an nn.Parameter in the graph
y.mul_(2)  # mutate it

assert n.args[0] in placeholders