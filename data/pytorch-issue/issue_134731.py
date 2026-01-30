import torch
from torch.utils._pytree import tree_map_only

# ===========
# Example 1
# ===========

class GroupedTensor:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __getitem__(self, index):
        mapped_attributes = tree_map_only(
            torch.Tensor,
            lambda x: x[index],
            (self.a, self.b),
        )
        return mapped_attributes

def getitem(grouped_tensor: GroupedTensor, index: torch.Tensor):
    return grouped_tensor[index]


grouped_tensor = GroupedTensor(torch.randn(4, 2, 5), torch.randn(4, 2,5,5))
compiled_fn = torch.compile(getitem, fullgraph=True)
compiled_fn(grouped_tensor, torch.tensor([2]))

# ===========
# Example 2
# ===========
def f(a, b, index):
    def call(index):
        mapped_attributes = tree_map_only(
            torch.Tensor,
            lambda x: x[index],
            (a, b),
        )
        return mapped_attributes
    return call(index)

a = torch.randn(4, 2, 5)
b = torch.randn(4, 2,5,5)
compiled_fn = torch.compile(f, fullgraph=True)
compiled_fn(a, b, torch.tensor([2]))