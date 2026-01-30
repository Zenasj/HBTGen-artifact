py
import torch


@torch.compile()
def test_foreach_add(a0, a1, b0, b1):
    return torch._foreach_add([a0, a1], [b0, b1])


print(
    test_foreach_add(
        torch.ones(10, 10, device="cuda"),
        torch.ones(20, 20, device="cpu"),
        torch.zeros(10, 10, device="cuda"),
        torch.zeros(20, 20, device="cpu"),
    )
)