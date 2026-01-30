import torch

class TestTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        x,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return super().__new__(cls, x, *args, **kwargs)


x = torch.ones(5)
test_tensor = TestTensor(x)