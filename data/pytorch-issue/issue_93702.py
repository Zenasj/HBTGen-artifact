import torch.nn as nn

import torch
import torchdynamo

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, inputs : torch.Tensor):
        mask1 = torch.ones((1, 16), device='cuda')
        mask2 = torch.full((1, 16), 3.0, device='cuda')
        out1  = inputs + mask1
        out2  = torch.nn.functional.softmax(out1, dim = -1)
        out3  = out2 - mask2
        out4  = self.dropout(out3); 
        return out4

mod = Repro().cuda()
opt_mod = torchdynamo.optimize("aot_nvfuser")(mod)
#opt_mod = torch.jit.script(mod)

args = [torch.randn(1, 2, 16, 16, device='cuda', requires_grad=True)]
grads = torch.randn(1, 2, 16, 16, device='cuda')

for _ in range(5) :
    args[0].grad = None
    out_res = opt_mod(*args)
    out_res.backward(grads)

class GraphModule(torch.nn.Module):
    def forward(self, primals_1):
        
        # Module stack: {}, File: simple_test.py:10, code: mask1 = torch.ones((1, 16), device='cuda')
        ones = torch.ops.aten.ones.default([1, 16], device = device(type='cuda'), pin_memory = False)
        alias = torch.ops.aten.alias.default(ones);  ones = None
        alias_1 = torch.ops.aten.alias.default(alias);  alias = None
        
        # Module stack: {}, File: simple_test.py:11, code: mask2 = torch.full((1, 16), 3.0, device='cuda')
        full = torch.ops.aten.full.default([1, 16], 3.0, device = device(type='cuda'), pin_memory = False)
        alias_2 = torch.ops.aten.alias.default(full);  full = None
        alias_3 = torch.ops.aten.alias.default(alias_2);  alias_2 = None
        
        # Module stack: {}, File: simple_test.py:12, code: out1  = inputs + mask1
        add = torch.ops.aten.add.Tensor(primals_1, alias_1);  primals_1 = alias_1 = None
        
        # Module stack: {}, File: simple_test.py:13, code: out2  = torch.nn.functional.softmax(out1, dim = -1)
        _softmax = torch.ops.aten._softmax.default(add, -1, False);  add = None
        
        # Module stack: {}, File: simple_test.py:14, code: out3  = out2 - mask2
        sub = torch.ops.aten.sub.Tensor(_softmax, alias_3);  alias_3 = None
        
        # Module stack: {'self_dropout': 'Dropout'}, File: simple_test.py:15, code: out4  = self.dropout(out3);
        native_dropout = torch.ops.aten.native_dropout.default(sub, 0.1, True);  sub = None
        getitem = native_dropout[0]
        getitem_1 = native_dropout[1];  native_dropout = None
        return [getitem, getitem_1, _softmax]

import torch
import torchdynamo

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

    def forward(self, inputs : torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor):
        out1  = inputs + mask1
        out2  = torch.nn.functional.softmax(out1, dim = -1)
        out3  = out2 - mask2
        out4  = self.dropout(out3); 
        return out4

mod = Repro().cuda()
opt_mod = torchdynamo.optimize("aot_nvfuser")(mod)
#opt_mod = torch.jit.script(mod)

args = [torch.randn(1, 2, 16, 16, device='cuda', requires_grad=True),
        torch.ones((1, 16), device='cuda'),
        torch.full((1, 16), 3.0, device='cuda')]

for _ in range(5) :
    out_res = opt_mod(*args)

class GraphModule(torch.nn.Module):                                                                          
    def forward(self, primals_1, primals_2, primals_3):                            
                                                                                                                            
        # Module stack: {}, File: simple_test.py:10, code: out1  = inputs + mask1                                           
        add = torch.ops.aten.add.Tensor(primals_1, primals_2);  primals_1 = primals_2 = None                     
                                                                                                                      
        # Module stack: {}, File: simple_test.py:11, code: out2  = torch.nn.functional.softmax(out1, dim = -1)            
        _softmax = torch.ops.aten._softmax.default(add, -1, False);  add = None                                            
                                                                                                                            
        # Module stack: {}, File: simple_test.py:12, code: out3  = out2 - mask2                                  
        sub = torch.ops.aten.sub.Tensor(_softmax, primals_3);  primals_3 = None                                       
                                                                                                                                      
        # Module stack: {'self_dropout': 'Dropout'}, File: simple_test.py:13, code: out4  = self.dropout(out3);    
        native_dropout = torch.ops.aten.native_dropout.default(sub, 0.1, True);  sub = None                        
        getitem = native_dropout[0]                                                                                        
        getitem_1 = native_dropout[1];  native_dropout = None                                                      
        return [getitem, getitem_1, _softmax]