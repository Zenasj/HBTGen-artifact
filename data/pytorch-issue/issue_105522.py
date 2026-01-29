# torch.rand(200, 64, 140, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_size=140, hidden_size=340, device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.param1 = nn.Parameter(torch.empty(input_size, 2 * hidden_size, device=device))
        self.param2 = nn.Parameter(torch.empty(hidden_size, 2 * hidden_size, device=device))

    def forward(self, input):
        batch_size = input.size(1)
        state1 = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        outs = []
        Wx = input @ self.param1
        Wx_inp, Wx_rec = torch.tensor_split(Wx, 2, dim=2)
        for wt_inp_t, wt_rec_t in zip(Wx_inp, Wx_rec):
            rec_mul = state1 @ self.param2
            rec_mul_inp, rec_mul_rec = torch.tensor_split(rec_mul, 2, dim=1)
            input_prev = wt_inp_t + rec_mul_inp
            output_gate = wt_rec_t + rec_mul_rec
            state1 = 1.0 + input_prev * torch.sigmoid(output_gate)
            outs.append(state1)
        outs = torch.stack(outs)
        return outs

def my_model_function():
    return MyModel(input_size=140, hidden_size=340, device='cuda')

def GetInput():
    return torch.rand(200, 64, 140, dtype=torch.float32)

