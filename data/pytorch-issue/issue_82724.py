# torch.rand(B, 1, dtype=torch.int64, device='cuda:0')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor_span):
        batch_size = tensor_span.shape[0]
        test_mask = torch.zeros((batch_size, 100, 100), dtype=torch.float32, device=tensor_span.device)
        for i in range(batch_size):
            test_mask[i, :tensor_span[i, 0], :tensor_span[i, 0]] = 1.0
        return test_mask

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 100, (1, 1), dtype=torch.int64, device='cuda:0')

