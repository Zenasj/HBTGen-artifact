import torch

def forward(self, L_ex_0_ : torch.Tensor, L_ex_1_ : torch.Tensor):
    l_ex_0_ = L_ex_0_
    l_ex_1_ = L_ex_1_

    # File: /data/users/ezyang/d/pytorch/test/inductor/test_torchinductor.py:4279 in fn, code: z = y.item() 
    item = l_ex_1_.item();  l_ex_1_ = None

    # No stacktrace found for following nodes
    _check_is_size = torch._check_is_size(item)
    _constrain_as_value = torch._constrain_as_value(item, 0, 0)

    # File: /data/users/ezyang/d/pytorch/test/inductor/test_torchinductor.py:4280 in fn, code: return torch.cat([x, x.new_ones(z)])
    new_ones = l_ex_0_.new_ones(item);  item = None
    cat = torch.cat([l_ex_0_, new_ones]);  l_ex_0_ = new_ones = None
    return (cat,)