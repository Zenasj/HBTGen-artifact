import torch.nn as nn
import torch.nn.functional as F

import torch                                                                  
import torchdynamo                                                            
from typing import List                                                       
                                                                              
                                                                              
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")                              
    gm.graph.print_tabular()                                                  
    return gm.forward  # return a python callable                             
                                                                              
                                                                              
def run():                                                                    
                                                                              
    counter = 0                                                               
                                                                              
    class TensorProxy(torch.Tensor):                                          
                                                                              
        @classmethod                                                          
        def __torch_function__(cls, func, types, args=(), kwargs=None):       
            nonlocal counter                                                  
            counter += 1                                                      
            return super().__torch_function__(func, types, args, kwargs)      
                                                                              
    def foo(x):                                                               
        # a good choice for initial testing is F.sigmoid, because             
        # it only has one argument                                            
        x = torch.nn.functional.sigmoid(x)                                    
        return x                                                              
                                                                              
    torchdynamo.config.traceable_tensor_subclasses.add(TensorProxy)           
    torchdynamo.config.debug = True                                           
    torchdynamo.config.trace = True                                           
    torchdynamo.config.normalize_ir = False                                   
    x = torch.randn(1).as_subclass(TensorProxy)                               
    foo(x)                                                                         
    with torchdynamo.optimize(my_compiler, nopython=True):                    
        foo(x)                                                                
      
                                                                        
run()