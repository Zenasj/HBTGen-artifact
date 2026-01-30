import torch
import torch.nn as nn

...
class Test(torch.nn.Module):
    __constants__ = ["flag"]
    def __init__(self):
        super().__init__()

        self.flag = ""
        self.activation = LinearActivation()
...

class Test(torch.nn.Module):
    flag: Final[str]
    def __init__(self): ...