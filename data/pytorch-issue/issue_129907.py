import torch

getitem = ...
sym_size_int = torch.ops.aten.sym_size.int(getitem, 0) # this is u0 = s0 + s1 
_check_is_size = torch._check_is_size(sym_size_int)
# we fail at this guy
sym_constrain_range_default = torch.ops.aten.sym_constrain_range.default(sym_size_int, min = 4, max = 1234)

# runtime assertion
add = sym_size_int + sym_size_int_1
eq = add == sym_size_int
_assert_scalar_default = torch.ops.aten._assert_scalar(eq, "Runtime assertion failed for expression Eq(s0 + s1, u0) on node 'eq'")