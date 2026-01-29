# torch.randint(10, (4,), dtype=torch.int64), torch.randint(10, (4,), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        grad, input_tensor = inputs
        # Compute on CPU
        result_cpu = torch.ops.aten.threshold_backward(grad, input_tensor, 1)
        # Compute on META
        grad_meta = grad.to('meta')
        input_meta = input_tensor.to('meta')
        result_meta = torch.ops.aten.threshold_backward(grad_meta, input_meta, 1)
        # Return boolean tensor indicating dtype match
        return torch.tensor(result_cpu.dtype == result_meta.dtype, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    grad = torch.randint(10, (4,), dtype=torch.int64)
    input_tensor = torch.randint(10, (4,), dtype=torch.int64)
    return (grad, input_tensor)

