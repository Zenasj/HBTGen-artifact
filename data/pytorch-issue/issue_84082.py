Python3
import torch

device='cuda'
dtype=torch.float32

a = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
x = torch.nested_tensor([a, b])

x.to_padded_tensor(0).sum().backward()

torch.testing.assert_close(a.grad, torch.ones(2, 4, device=device, dtype=dtype))
torch.testing.assert_close(b.grad, torch.ones(5, 4, device=device, dtype=dtype))

Python
device='cuda'
dtype=torch.float32

a = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
x = torch.nested_tensor([a, b]).requires_grad_()

x.to_padded_tensor(0).sum().backward()

print(f"x's grad is {x.grad}")