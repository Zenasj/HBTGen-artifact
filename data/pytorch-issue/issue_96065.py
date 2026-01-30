import torch

def test_grad(alpha, device):
    other = torch.tensor([1.], device=device)
    input = torch.tensor([1.], device=device, requires_grad=True)

    with torch.no_grad():
        input.add_(other, alpha=alpha)

    assert input.requires_grad

test_grad(alpha=1, device="cpu")
test_grad(alpha=0, device="cpu")
test_grad(alpha=1, device="mps")
test_grad(alpha=0, device="mps") # assertion fails with >=1.12.1