py
import torch
a = torch.rand(10000, 10000)
@torch.compile(backend='aot_eager')
def forward(arg0_1):
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 2)
    sum_1 = torch.ops.aten.sum.dim_IntList(pow_1, [1]);  pow_1 = None
    _tensor_constant0 = torch.tensor([1])
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    cumsum = torch.ops.aten.cumsum.default(lift_fresh_copy, 0);  lift_fresh_copy = None
    slice_1 = torch.ops.aten.slice.Tensor(cumsum, 0, 0, -1);  cumsum = None
    neg = torch.ops.aten.neg.default(slice_1);  slice_1 = None
    unbind = torch.ops.aten.unbind.int(neg);  neg = None
    new_zeros = torch.ops.aten.new_zeros.default(sum_1, [10000, 1, 1], pin_memory = False);  sum_1 = None
    diagonal = torch.ops.aten.diagonal.default(new_zeros, 0, 1, 2)
    fill_ = torch.ops.aten.fill_.Scalar(diagonal, 1);  diagonal = None
    view = torch.ops.aten.view.default(new_zeros, [10000, 1]);  new_zeros = None
    unsqueeze = torch.ops.aten.unsqueeze.default(view, 2);  view = None
    view_1 = torch.ops.aten.view.default(unsqueeze, [10000, 1, 1]);  unsqueeze = None
    view_2 = torch.ops.aten.view.default(view_1, [10000, 1, 1]);  view_1 = None
    expand = torch.ops.aten.expand.default(view_2, [10000, 1, 10000]);  view_2 = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(arg0_1, 1.0);  arg0_1 = None
    mul = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
    view_3 = torch.ops.aten.view.default(mul, [10000, 1, 10000]);  mul = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, view_3);  expand = view_3 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(mul_1, [1], 1);  mul_1 = None
    getitem = split_with_sizes[0];  split_with_sizes = None
    view_4 = torch.ops.aten.view.default(getitem, [10000, 10000]);  getitem = None
    return view_4

forward(a)