import torch
J = torch.eye(2).unsqueeze(0).expand(5, 2, 2)
for i in range(2):
    J[:, i, :] = torch.randn([5, 2])

import torch
J = torch.eye(2).unsqueeze(0).expand(5, 2, 2)
# completely new tensor allocated by cloning
J_clone = J.clone()
for i in range(2):
    J_clone[:, i, :] = torch.randn([5, 2])
print(J_clone)

import torch
J = torch.ones((5,2,2))
for i in range(2):
    J[:, i, :] = torch.randn([5, 2])
print(J)