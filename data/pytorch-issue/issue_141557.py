import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.dropout(x, p=0.5)  # set training=False can avoid the issue
        return x


with torch.no_grad():
    torch.manual_seed(SEED)
    model = Model().eval().cuda()
    torch.manual_seed(SEED)
    c_model = torch.compile(model).cuda()

    torch.manual_seed(SEED)
    input_tensor = torch.randn(1, 2)
    inputs = [input_tensor]

    for i in range(10):
        print(f"round {i}")
        torch.manual_seed(SEED)
        print(model(*inputs))

    torch.manual_seed(SEED)
    c_output = c_model(*inputs)
print(f"compiled_output\n{c_output}")