# torch.rand(B, C, H, dtype=torch.float32)  # Inferred input shape: (10, 212, 500000)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        p_list, i_list = [], []
        for batch_item in x:
            topk_probs, topk_ids = batch_item.topk(20, sorted=False, dim=-1)
            p_list.append(topk_probs)
            i_list.append(topk_ids)
        p_ = torch.stack(p_list, dim=0)
        i_ = torch.stack(i_list, dim=0)
        return p_, i_

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(size=[10, 212, 500000], dtype=torch.float32).to('cuda:0')

