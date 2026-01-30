import torch

def fn(x):
    assert x.isfinite().all()
    return x * x

def forward(self, L_x_ : torch.Tensor):
     l_x_ = L_x_
     
     # File: <ipython-input-2-e6a7009c79f6>:9, code: assert x.isfinite().all()
     isfinite = l_x_.isfinite()
     all_1 = isfinite.all();  isfinite = None
     _assert_async = torch._assert_async(all_1, 'assertion error');  all_1 = None
     
     # File: <ipython-input-2-e6a7009c79f6>:10, code: return x * x
     mul = l_x_ * l_x_;  l_x_ = None
     return (mul,)