import torch
import torch._dynamo.testing
import torch._inductor.ops_handler
import torch._inductor.utils
import torch._inductor
import torch._dynamo

def test_repro():
 
     def fn_2(x):
         # constrain in two directions
         if x.shape[0] > 5:
             return x.cos()
         if x.shape[0] < 5:
             return x * 2
         # x.shape[0] == 5 at this point
         return x.sin()

     torch._dynamo.reset()
     _x = torch.randn([5, 3, 3])
     torch._dynamo.mark_dynamic(_x, 0)
     torch.compile(backend="inductor", dynamic=None)(fn_2)(_x)

test_repro()