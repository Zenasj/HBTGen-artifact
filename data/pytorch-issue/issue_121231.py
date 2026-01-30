import torch.nn as nn

py
import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.w1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device='cuda')
        self.b1 = torch.zeros(3, device='cuda')
        self.w2 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device='cuda')
        self.b2 = torch.tensor([[-1.0, -1.0, -1.0]], device='cuda')
        self.w3 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device='cuda')
        self.b3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')

    def forward(self, x):
        out1 = torch.nn.functional.linear(x, self.w1, self.b1)
        out2 = torch.nn.functional.linear(x, self.w2, self.b2)
        out3 = torch.nn.functional.linear(x, self.w3, self.b3)
        return (out1, out2, out3)


func = Model().to('cuda').eval()

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).cuda()

with torch.no_grad():
    print(func(x.clone()))

    func1 = torch.compile(func)
    print(func1(x.clone()))