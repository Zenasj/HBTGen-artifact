import torch.nn as nn

import torch
from torch.export import export_for_training

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.linear(x)
        x = torch.nn.functional.tanh(x)
        return x

if __name__ == "__main__":
    with torch.no_grad():
        m = Model().eval()
        shape = (2, 3)
        x = torch.randn(*shape).to(torch.bfloat16)
        x2 = torch.randn(*shape)
        exported_model = export_for_training(
            m,
            (x,),
        ).module()
        print("exported_model is: {}".format(exported_model), flush=True)
        cfn = torch.compile(exported_model)
        cfn(x2)