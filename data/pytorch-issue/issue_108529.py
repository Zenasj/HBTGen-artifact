from typing import Tuple, Any
import torch
import torch.autograd


class TestFunction(torch.autograd.Function):
    @staticmethod
    def forward(tensor: torch.Tensor, var1: int = 1, var2: int = 2) -> torch.Tensor:
        print(f"tensor: {tensor}, var1: {var1}, var2: {var2}")
        return tensor

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any], output: Any) -> Any:
        print(f"inputs are {inputs}")
        tensor, var1, var2 = inputs

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        grad = grad_output[0]

        return grad, None, None

if __name__ == "__main__":
    tensor = torch.tensor(1.0, requires_grad=True)
    output = TestFunction.apply(tensor, 3)
    output.backward()