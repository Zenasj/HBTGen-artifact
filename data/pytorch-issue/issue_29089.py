# torch.rand(C_out, C_in, H, W, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, tensor):
        fan_in_orig, fan_out_orig = self._calculate_original(tensor)
        fan_in_fixed, fan_out_fixed = self._calculate_fixed(tensor)
        return torch.tensor(fan_in_orig == fan_in_fixed and fan_out_orig == fan_out_fixed, dtype=torch.bool)
    
    def _calculate_original(self, tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        if dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:  # Original redundant condition
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out
    
    def _calculate_fixed(self, tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        if dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = tensor[0][0].numel()  # Fixed version removes redundant check
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

def my_model_function():
    return MyModel()

def GetInput():
    # 4D tensor example for convolution weights (out_channels, in_channels, kernel_h, kernel_w)
    return torch.rand(3, 2, 5, 5, dtype=torch.float)

