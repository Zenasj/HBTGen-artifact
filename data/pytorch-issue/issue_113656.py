import torch
ts = [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2], [3, 4]])]
nested = torch.nested.nested_tensor(ts)

print(ts[0] is ts[1])  # False
print(id(ts[0]) == id(ts[1]))  # False

print(nested[0] is nested)  # False
print(id(nested[0]) == id(nested))  # False

print(nested[0] is nested[1])  # False
print(id(nested[0]) == id(nested[1]))  # True! What??