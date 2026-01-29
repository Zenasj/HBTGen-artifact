# Input is a tuple of tensors with shapes: ( (768,), (768,), (768,), (768,), (768,), (768,), (8, 576, 768), (8, 1, 768), (8, 768), (8, 577, 1), (8, 577, 1), (768, 768), (128, 1, 577), (768, 768) )
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = inputs

        # First part
        view = arg8_1.view(8, 1, 768)
        mul = arg0_1 * view
        add = arg7_1 + mul
        cat = torch.cat([add, arg6_1], dim=1)
        sub = cat - arg9_1
        mul_1 = sub * arg10_1
        mul_2 = mul_1 * arg2_1
        add_1 = mul_2 + arg3_1
        view_1 = add_1.view(4616, 768)
        addmm = torch.addmm(arg4_1, view_1, arg11_1)
        view_2 = addmm.view(8, 577, 768)
        view_3 = view_2.view(8, 577, 16, 48)
        permute = view_3.permute(0, 2, 1, 3)
        expand = permute.expand(permute.size())
        clone = expand.clone(memory_format=torch.contiguous_format)
        view_4 = clone.view(128, 577, 48)
        bmm = torch.bmm(arg12_1, view_4)
        view_5 = bmm.view(8, 16, 1, 48)
        permute_1 = view_5.permute(0, 2, 1, 3)
        view_6 = permute_1.view(8, 1, 768)
        view_7 = view_6.view(8, 768)
        addmm_1 = torch.addmm(arg5_1, view_7, arg13_1)
        view_8 = addmm_1.view(8, 1, 768)
        mul_3 = arg1_1 * view_8
        add_2 = add + mul_3

        return add_2

def my_model_function():
    return MyModel()

def GetInput():
    tensors = [
        torch.rand(768, dtype=torch.float32),          # arg0_1
        torch.rand(768, dtype=torch.float32),          # arg1_1
        torch.rand(768, dtype=torch.float32),          # arg2_1
        torch.rand(768, dtype=torch.float32),          # arg3_1
        torch.rand(768, dtype=torch.float32),          # arg4_1
        torch.rand(768, dtype=torch.float32),          # arg5_1
        torch.rand(8, 576, 768, dtype=torch.float32),  # arg6_1
        torch.rand(8, 1, 768, dtype=torch.float32),    # arg7_1
        torch.rand(8, 768, dtype=torch.float32),       # arg8_1
        torch.rand(8, 577, 1, dtype=torch.float32),    # arg9_1
        torch.rand(8, 577, 1, dtype=torch.float32),    # arg10_1
        torch.rand(768, 768, dtype=torch.float32),     # arg11_1
        torch.rand(128, 1, 577, dtype=torch.float32),  # arg12_1
        torch.rand(768, 768, dtype=torch.float32),     # arg13_1
    ]
    return tuple(tensors)

