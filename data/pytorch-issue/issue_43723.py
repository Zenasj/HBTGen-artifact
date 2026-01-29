# (torch.randint(0, 3, (3,)), torch.tensor([0, 2], dtype=torch.long))  # Input shape: index (3,) and offsets (2,)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(
            [[0., 1., 2., 3.],
             [4., 5., 6., 7.],
             [8., 9., 10., 11.]]
        ))  # Original weight tensor (contiguous)

    def forward(self, inputs):
        index, offsets = inputs
        # Non-contiguous weight slice (replicates issue scenario)
        non_contig_weight = self.weight[:, :3]  
        # Contiguous version for comparison
        contig_weight = non_contig_weight.contiguous()  

        # Compute outputs for both cases
        output_non_contig = F.embedding_bag(
            index, non_contig_weight, offsets, mode="mean"
        )
        output_contig = F.embedding_bag(
            index, contig_weight, offsets, mode="mean"
        )

        # Return boolean tensor indicating if outputs differ significantly
        are_close = torch.allclose(output_non_contig, output_contig, atol=1e-5)
        return torch.tensor(are_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicates the exact input tensors from the issue's example
    index = torch.tensor([0, 1, 2], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    return (index, offsets)

