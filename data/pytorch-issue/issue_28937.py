import torch

3
leaf_tensor=torch.randn(3)
x=leaf_tensor.tanh()
optimizer = optim.Adam([leaf_tensor], lr=0.1)

for epoch in range(30):
    loss=x.sum()
    loss.backward()
    optimizer.step()
    x=leaf_tensor.tanh()