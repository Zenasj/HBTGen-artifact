import torch

# test without view
X = torch.tensor([[[0.25]],[[ 0.75]]],requires_grad=True,)
print(f"X.shape: {X.shape}")
X.sum().backward()
print(f"X.grad: {X.grad}")

# test with view
X_view = torch.tensor([0.25, 0.75], requires_grad=True,).view(2, 1, 1)
print(f"X_view.shape: {X_view.shape}")
X_view.sum().backward()
print(f"X_view.grad: {X_view.grad}")
print(f"X_view.grad is None: {X_view.grad is None}")

X0 = torch.tensor([0.25, 0.75], requires_grad=True,)
X_view = X0.view(2, 1, 1)
print(f"X_view.shape: {X_view.shape}")
X_view.sum().backward()
print(f"X_view.grad: {X_view.grad}")
print(f"X_view.grad is None: {X_view.grad is None}")
print(f"X0.grad: {X0.grad}")