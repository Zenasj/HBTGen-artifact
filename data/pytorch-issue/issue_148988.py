# torch.rand(20, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1g = None  # Stores the detached tensor for gradient computation

    def forward(self, base_tensor):
        t1 = base_tensor.view(5, 4)
        t2 = t1 * 10  # Compute key/value tensor

        # Create nested tensors for query (n1) and key/value (n2)
        n1_data = [t1[:2], t1[2:5]]  # Split into [2x4, 3x4]
        n1 = torch.nested.as_nested_tensor(n1_data, layout=torch.jagged)
        self.n1g = n1.clone().detach().requires_grad_()  # Detached leaf tensor for gradients

        n2_data = [t2[:1], t2[1:5]]  # Split into [1x4, 4x4]
        n2 = torch.nested.as_nested_tensor(n2_data, layout=torch.jagged)

        # Prepare for scaled_dot_product_attention
        query = self.n1g.unsqueeze(2).transpose(1, 2)
        key = value = n2.unsqueeze(2).transpose(1, 2)
        output = F.scaled_dot_product_attention(query, key, value)
        loss = output.values().sum()  # Sum over all elements of nested tensor values
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    return torch.arange(20).float().requires_grad_()

