from typing import Optional

import torch

def triu_onnx(inputs: torch.FloatTensor,
              diagonal: Optional[int] = 0) -> torch.FloatTensor:
    """Caveat to export an triu-based operator with ONNX.

    Args:
        inputs: Input tensor.
        diagonal: Value of diagonal.

    Returns:
        (torch.FloatTensor): Output tensor.

    """

    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) <= (arange2 - diagonal)

    return inputs.masked_fill(mask == 0, 0)


def tril_onnx(inputs: torch.FloatTensor,
              diagonal: Optional[int] = 0) -> torch.FloatTensor:
    """Caveat to export an tril-based operator with ONNX.

    Args:
        inputs: Input tensor.
        diagonal: Value of diagonal.

    Returns:
        (torch.FloatTensor): Output tensor.

    """

    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)

    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) >= (arange2 - diagonal)

    return inputs.masked_fill(mask == 0, 0)


if __name__ == '__main__':
    inputs = torch.randn((10, 10))
    assert (triu_onnx(inputs) - torch.triu(inputs)).sum() == 0
    assert (tril_onnx(inputs) - torch.tril(inputs)).sum() == 0