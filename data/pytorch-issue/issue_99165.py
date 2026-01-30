from typing import Optional

import torch


def node_hook(
    grad_inputs: tuple[torch.Tensor],
    grad_outputs: tuple[torch.Tensor],
) -> Optional[tuple[torch.Tensor]]:
  print(
      f"grad_inputs: {grad_inputs}\n"
      f"grad_outputs: {grad_outputs}"
  )
  # Returning `grad_inputs` work, but returning `grad_outputs` fails.
  return grad_inputs
  # return grad_outputs  # RuntimeError: hook 'node_hook' has returned an incorrect number of values (got 1, but expected 2)


a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)
c = a + b
# c.grad_fn is AddBackward0 which has two inputs and one output.
c.grad_fn.register_hook(node_hook)
c.backward()