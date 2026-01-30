import torch                                                                                                                                                  
from torch.testing._internal.jit_utils import disable_autodiff_subgraph_inlining                                        
                                                                                                                        
torch._C._jit_set_profiling_mode(False)                                                                                 
torch._C._jit_set_profiling_executor(False)                                                                             
                                                                                                                        
x = torch.randn(1, 4).requires_grad_()                                                                                  
with disable_autodiff_subgraph_inlining(False):                                                                         
                                                                                                                        
  def f(x):                                                                                                             
    o = x + 1.0                                                                                                         
    split_o = torch.split(o, 2, dim=1)                                                                                  
                                                                                                                        
    return o, split_o[0], split_o[1]                                                                                    
                                                                                                                        
  script_f = torch.jit.script(f)                                                                                        
                                                                                                                        
  o = f(x)                                                                                                              
  torch.cat(o, dim=1).sum().backward()                                                                                  
                                                                                                                        
  print(x.grad)                                                                                                         
                                                                                                                        
  jit_o = script_f(x)                                                                                                   
  torch.cat(jit_o, dim=1).sum().backward()                                                                              
  jit_o = script_f(x)                                                                                                   
  torch.cat(jit_o, dim=1).sum().backward()                                                                              
  jit_o = script_f(x)                                                                                                   
  torch.cat(jit_o, dim=1).sum().backward()                                                                              
                                                                                                                        
  x.grad.zero_()                                                                                                        
  jit_o = script_f(x)                                                                                                   
  torch.cat(jit_o, dim=1).sum().backward()                                                                              
                                                                                                                        
  print(x.grad)                                                                                                         
                                                                                                                        
  print(script_f.graph_for(x))