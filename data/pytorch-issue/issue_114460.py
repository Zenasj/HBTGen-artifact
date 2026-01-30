## Baseline
print(1 in {1, 2})  # True
print(1 in [1, 2])  # True
print(1 in (1, 2))  # True
## Tests
print("-----------------")
print(1 in {tensor(1), 2})  # False
print(1 in [tensor(1), 2])  # True
print(1 in (tensor(1), 2))  # True
print("-----------------")
print(tensor(1) in {tensor(1), 2})  # False
print(tensor(1) in (tensor(1), 2))  # True
print(tensor(1) in [tensor(1), 2])  # True
print("-----------------")
print(tensor(1) in {1, 2})  # False
print(tensor(1) in (1, 2))  # True
print(tensor(1) in [1, 2])  # True

import torch

a = torch.tensor(1)
b = hash(a)
c = {b}
print(a in c)  # False
for i in c:
    print(hash(i) == hash(a))  # True