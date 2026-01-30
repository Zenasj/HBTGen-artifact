import torch.nn as nn

from typing import Dict

import torch
from torch import nn
from torch.optim.adamw import AdamW

def assert_state_dict_equal(first: Dict[str, torch.Tensor], second: Dict[str, torch.Tensor]):
    for name, first_tensor in first.items():
        second_tensor = second[name]
        torch.testing.assert_close(first_tensor, second_tensor, atol=0, rtol=0)

def test_empty_tensor_in_optimizer(device: torch.device):
    model = nn.Linear(2, 3, bias=True, device=device)
    reference_model = nn.Linear(2, 3, bias=True, device=device)

    # sync parameters
    with torch.no_grad():
        model.weight.copy_(reference_model.weight)
        model.bias.copy_(reference_model.bias)

    empty_tensor = torch.tensor([], requires_grad=True, device=device)

    optimizer = AdamW(
        [
            model.weight,
            model.bias,
            empty_tensor,
        ]
    )
    reference_optimizer = AdamW(reference_model.parameters())

    random_input = torch.randn(5, 2, device=device)

    # Check that state dict are equal before optimizer step
    assert_state_dict_equal(model.state_dict(), reference_model.state_dict())

    model(random_input).sum().backward()
    empty_tensor.grad = torch.tensor([], requires_grad=False, device=device) # important
    reference_model(random_input).sum().backward()

    optimizer.step()
    reference_optimizer.step()

    # Check that state dict are equal after optimizer step
    assert_state_dict_equal(model.state_dict(), reference_model.state_dict())


def main():
    for device in [
        torch.device("cpu"),
        torch.device("cuda")
    ]:
        print(f"Testing device: {device}")
        test_empty_tensor_in_optimizer(device)
        print("Pass")


if __name__ == "__main__":
    main()