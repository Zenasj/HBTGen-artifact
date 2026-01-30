import random

import torch

@torch.compile()
def f():                                                                                                                                                                                                                              
    device = 'cpu'
    tensor = torch.zeros((1,), dtype=torch.double, device=device)                                                                                                                                                                     
    index = torch.tensor([0], dtype=torch.long, device=device)                                                                                                                                                                                        
    source = torch.rand((1,), dtype=torch.double, device=device)                                                                                                                                                                      
    out = tensor.index_add(0, index, source, alpha=2.) / 2                                                                                                                                                                            
    return source.item(), out.item()

torch.random.manual_seed(0)
print(f())  # prints (0.08809276670217514, 0.0)

def forward(self):
    # File: /home/jezng/tmp/test.py:6, code: tensor = torch.zeros((1,), dtype=torch.double, device=device)
    full: f64[1] = torch.ops.aten.full.default([1], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

    # File: /home/jezng/tmp/test.py:7, code: index = torch.tensor([0], dtype=torch.long, device=device)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: i64[1] = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None

    # File: /home/jezng/tmp/test.py:8, code: source = torch.rand((1,), dtype=torch.double, device=device)
    rand: f64[1] = torch.ops.aten.rand.default([1], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)

    # File: /home/jezng/tmp/test.py:9, code: out = tensor.index_add(0, index, source, alpha=2.) / 2
    mul: f64[1] = torch.ops.aten.mul.Tensor(rand, 2.0)
    index_put: f64[1] = torch.ops.aten.index_put.default(full, [lift_fresh_copy], mul, True);  full = lift_fresh_copy = mul = None
    div: f64[1] = torch.ops.aten.div.Tensor(index_put, 2);  index_put = None
    return (rand, div)