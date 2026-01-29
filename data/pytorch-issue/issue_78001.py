# torch.rand(100, 100), torch.randint(0, 2, (4,)), torch.randint(0, 2, (4,))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs):
        tensor_2d, labels, ref_labels = inputs
        # First test: nonzero on non-contiguous tensor
        nonzeros = tensor_2d.nonzero()  # Triggers MPS issue
        
        # Second test: matches/diffs calculation and torch.where
        matches, diffs = self.get_matches_and_diffs(labels, ref_labels)
        a2_idx, n_idx = torch.where(diffs)  # Triggers MPS warning
        
        return nonzeros, a2_idx, n_idx
    
    def get_matches_and_diffs(self, labels, ref_labels):
        if ref_labels is None:
            ref_labels = labels
        labels1 = labels.unsqueeze(1)
        labels2 = ref_labels.unsqueeze(0)
        matches = (labels1 == labels2).to(torch.uint8)  # .byte() deprecated
        diffs = matches ^ 1
        if ref_labels is labels:
            matches.fill_diagonal_(0)
        return matches, diffs

def my_model_function():
    return MyModel()

def GetInput():
    tensor_2d = torch.rand(100, 100, dtype=torch.float32)
    labels = torch.randint(0, 2, (4,), dtype=torch.long)
    ref_labels = torch.randint(0, 2, (4,), dtype=torch.long)
    return (tensor_2d, labels, ref_labels)

