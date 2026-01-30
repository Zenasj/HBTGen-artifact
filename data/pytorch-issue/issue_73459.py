import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(MyTensor(...))
        self.register_parameter('param2', nn.Parameter(MyTensor(...)))
    ...