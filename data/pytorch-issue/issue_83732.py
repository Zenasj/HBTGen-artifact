import torch
import torch.nn as nn
# Example of target with class probabilities
loss1 = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
print(input)
print(target)
output = loss1(input, target)
print(output)
output.backward()
print(output)

import torch
import torch.nn as nn
# Example of target with class probabilities
loss1 = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
target = torch.argmax(target, dim=1)  # Might need it! 
print(input)
print(target)
output = loss1(input, target)
print(output)
output.backward()
print(output)