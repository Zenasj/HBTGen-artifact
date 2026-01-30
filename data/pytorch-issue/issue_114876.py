import torch

nested_a = torch.nested.nested_tensor([
    torch.randn(2, 5),
    torch.randn(3, 5)])

nested_b = torch.nested.nested_tensor([
    torch.randn(3, 5),
    torch.randn(4, 5)])

# Concatenate nested_a and nested_b should result
# in a nested_tensor with shape [(5, 5), (7, 5)]
nested_ab = torch.cat(
    [nested_a, nested_b], dim=0)