# torch.rand(4,512,768, dtype=torch.float), torch.rand(4,512,768, dtype=torch.float), torch.rand(768, dtype=torch.float), torch.rand(768, dtype=torch.float)
import torch
import torch.nn as nn

class Fusion7(nn.Module):
    def forward(self, T0, T1, T2, T3):
        T4 = T0 + T1
        T5 = T2.unsqueeze(0).unsqueeze(0).expand(4, 512, -1)
        T6 = T3.unsqueeze(0).unsqueeze(0).expand(4, 512, -1)
        T7, T8 = torch.var_mean(T4, unbiased=False, dim=2, keepdim=False)
        T9 = T7.unsqueeze(-1)
        T10 = T8.unsqueeze(-1)
        S11 = 1e-12
        T12 = T9 + S11
        T14 = torch.rsqrt(T12)
        T13 = T10.expand(-1, -1, 768)
        T15 = T4 - T13
        T16 = T14.expand(-1, -1, 768)
        T17 = T15 * T16
        T18 = T17 * T5
        T19 = T18 + T6
        T20 = T19
        return T4, T10, T14, T20

class Fusion10(nn.Module):
    def forward(self, T0, T1, T2, T3):
        T4 = T0.unsqueeze(0).unsqueeze(0).expand(4,512,-1)
        T5 = T1 + T2
        T6 = T3.unsqueeze(0).unsqueeze(0).expand(4,512,-1)
        T7, T8 = torch.var_mean(T5, unbiased=False, dim=2, keepdim=False)
        T9 = T7.unsqueeze(-1)
        T10 = T8.unsqueeze(-1)
        S11 = 1e-12
        T12 = T9 + S11
        T14 = torch.rsqrt(T12)
        T13 = T10.expand(-1, -1, 768)
        T15 = T5 - T13
        T16 = T14.expand(-1, -1, 768)
        T17 = T15 * T16
        T18 = T17 * T6
        T19 = T18 + T4
        T20 = T19
        return T5, T10, T14, T20

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion7 = Fusion7()
        self.fusion10 = Fusion10()
    
    def forward(self, inputs):
        a, b, c, d = inputs
        outputs7 = self.fusion7(a, b, c, d)
        outputs10 = self.fusion10(c, a, b, d)
        for o7, o10 in zip(outputs7, outputs10):
            if not torch.allclose(o7, o10, atol=1e-5, rtol=1e-5):
                return torch.tensor(False, dtype=torch.bool)
        return torch.tensor(True, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.rand(4, 512, 768, dtype=torch.float)
    b = torch.rand(4, 512, 768, dtype=torch.float)
    c = torch.rand(768, dtype=torch.float)
    d = torch.rand(768, dtype=torch.float)
    return (a, b, c, d)

