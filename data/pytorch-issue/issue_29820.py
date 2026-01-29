# torch.rand(3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, input):
        sz = input.shape[0]
        # Generate old (buggy) mask without transpose
        mask_old = torch.triu(torch.ones(sz, sz)) == 1
        mask_old = mask_old.float().masked_fill(mask_old == 0, float('-inf')).masked_fill(mask_old == 1, 0.0)
        
        # Generate new (fixed) mask with transpose
        mask_new = torch.triu(torch.ones(sz, sz)) == 1
        mask_new = mask_new.transpose(0, 1)
        mask_new = mask_new.float().masked_fill(mask_new == 0, float('-inf')).masked_fill(mask_new == 1, 0.0)
        
        # Expected mask (upper triangle -inf)
        expected = torch.zeros(sz, sz)
        expected.triu(1).fill_(-float('inf'))
        
        # Compare both masks against expected
        old_correct = torch.allclose(mask_old, expected)
        new_correct = torch.allclose(mask_new, expected)
        return torch.tensor([not old_correct, new_correct], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

