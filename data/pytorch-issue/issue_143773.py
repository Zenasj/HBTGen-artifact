import torch.nn as nn

import torch

@torch.library.custom_op("mylib::custom_add", mutates_args=())
def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def custom_add_direct(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

foo_lib = torch.library.Library("foo", "FRAGMENT")

def direct_register_custom_op(
    op_name,
    op_func,
    mutates_args
):
    schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    foo_lib.define(op_name + schema_str)
    foo_lib.impl(op_name, op_func, "CUDA")

direct_register_custom_op("foo::custom_add", custom_add_direct, mutates_args=())

# Create a module that uses the custom operator                                                                       
# class CustomModule(torch.nn.Module):
#     def forward(self, x, y):
#         # Same result with decorator and direct registration, when jit.loaded standalone:                             
#         # Unknown builtin op: foo::custom_add.                                                                        
#         return torch.ops.mylib.custom_add(x, y)
#         # return torch.ops.foo.custom_add(x, y)                                                                       

# # Create an instance and save it                                                                                      
# module = CustomModule()
# example_input1 = torch.randn(3, 4).cuda()
# example_input2 = torch.randn(3, 4).cuda()
# traced_module = torch.jit.trace(module, (example_input1, example_input2))
# traced_module.save("custom_module.pt")
# This works here, but fails standalone, in both Python and C++:                                                      
traced_module = torch.jit.load("custom_module.pt")
# out = traced_module(example_input1, example_input2)
# print(out)