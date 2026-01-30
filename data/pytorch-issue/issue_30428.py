import torch.nn as nn

class WrapRPN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        # type: (Dict[str, Tensor]) -> int
        return 0