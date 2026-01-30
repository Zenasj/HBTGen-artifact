import torch

@torch.compile
def f(x, y):
    return x[y]


x = torch.randn(1, device="cuda")
y = torch.ones(7, device="cuda")

def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):                                                                                                                                              
    xnumel = 7                                                                                                                                                                                                       
    xoffset = tl.program_id(0) * XBLOCK                                                                                                                                                                              
    xindex = xoffset + tl.arange(0, XBLOCK)[:]                                                                                                                                                                       
    xmask = xindex < xnumel                                                                                                                                                                                          
    x0 = xindex                                                                                                                                                                                                      
    tmp0 = tl.load(in_ptr0 + (0))                                                                                                                                                                                    
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])                                                                                                                                                                           
    tmp2 = tl.load(in_ptr1 + (0))                                                                                                                                                                                    
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])                                                                                                                                                                           
    tl.store(out_ptr0 + (tl.broadcast_to(x0, [XBLOCK])), tmp3, xmask)