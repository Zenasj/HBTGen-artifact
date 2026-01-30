from torchinductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

inps = [(torch.Size([4, 359613, 4]), torch.float32), (torch.Size([4, 5000]), torch.int64), (torch.Size([4, 1]), torch.int64), (torch.Size([4, 5000]), torch.float32)]
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]



def forward(self, arg0_1, cat_default_1, unsqueeze_default, index_tensor):
    index_tensor_2 = torch.ops.aten.index.Tensor(arg0_1, [unsqueeze_default, cat_default_1]);  arg0_1 = unsqueeze_default = cat_default_1 = None
    return []
forward = make_fx(forward)(*inps)
compile_fx_inner(forward, inps)

import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torchinductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

inps = [(torch.Size([]), torch.int64, 'cuda'), (torch.Size([37]), torch.int32, 'cuda'), (torch.Size([1406, 1]), torch.int32, 'cuda'), (torch.Size([1406, 1]), torch.int32, 'cuda'), (torch.Size([1406, 1]), torch.int32, 'cuda')]
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype, device) in inps]



def forward(self, fill__scalar_6, arange_7, unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14):
    mul_tensor_7 = torch.ops.aten.mul.Tensor(arange_7, fill__scalar_6);  arange_7 = fill__scalar_6 = None
    view_default_15 = torch.ops.aten.view.default(mul_tensor_7, [-1, 1]);  mul_tensor_7 = None
    expand_default_6 = torch.ops.aten.expand.default(view_default_15, [37, 38]);  view_default_15 = None
    clone_default_7 = torch.ops.aten.clone.default(expand_default_6, memory_format = torch.contiguous_format);  expand_default_6 = None
    _unsafe_view_default_7 = torch.ops.aten._unsafe_view.default(clone_default_7, [1406]);  clone_default_7 = None
    unsqueeze_default_15 = torch.ops.aten.unsqueeze.default(_unsafe_view_default_7, 1);  _unsafe_view_default_7 = None
    cat_default_3 = torch.ops.aten.cat.default([unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, unsqueeze_default_15], 1);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = unsqueeze_default_15 = None
    return (cat_default_3,)


compile_fx_inner(make_fx(forward)(*inps), inps)

from torchinductor.compile_fx import compile_fx_inner
from torch.fx.experimental.proxy_tensor import make_fx

inps = [(torch.Size([3, 4]), torch.float32, 'cpu'), (torch.Size([4, 359613, 4]), torch.float16, 'cuda')]
inps = [torch.zeros(())] + [torch.ones(shape, dtype=dtype, device=device) for (shape, dtype, device) in inps]



def forward(self, arg15_1, view_default_40):
    _to_copy_default_4 = torch.ops.aten._to_copy.default(arg15_1, dtype = torch.float16, device = device(type='cuda', index=0));  arg15_1 = None
    return [view_default_40]

compile_fx_inner(make_fx(forward)(*inps), inps)

In [10]: utils.suggest_memory_format(torch.as_strided(arg0_1, (1, 3, 427, 640), (0, 1, 1920, 3)))
Out[10]: torch.contiguous_format