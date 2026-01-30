import torch.nn as nn

import torch
# enable RNN in torchdynamo
from torch._dynamo import config
config.allow_rnn = True

device = "cuda"
x = torch.zeros([1, 8, 640], device=device)
h = (torch.zeros([2, 8, 640], device=device), torch.zeros([2, 8, 640], device=device))
d1 = torch.zeros(1, device="cpu", pin_memory=True)
d2 = torch.zeros(1, device="cpu", pin_memory=True)
d3 = torch.zeros(1, device="cpu", pin_memory=True)
d4 = torch.zeros(1, device="cpu", pin_memory=True)

class MY_LSTM(torch.nn.Module):
    def __init__(self):
        super(MY_LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=640, hidden_size=640, num_layers=2, dropout=0.2, proj_size=0
        )

    def forward(self, x, h):
        g, hid = self.lstm(x, h)

        # copy the "intermediate shape" of hid
        d1.copy_(hid[0].shape[0])
        d2.copy_(hid[0].shape[1])
        d3.copy_(hid[0].shape[2])
        if hid[0].dim() > 3:
            d4.copy_(hid[0].shape[3])

        return g, hid

lstm = MY_LSTM().to(device=device)
compiled_lstm = torch.compile(lstm, backend="eager")

print("\n====================== Eager PyTorch ======================")
_, hid_ret = lstm(x, h)

print("Return shape of hid[0]:")
print(hid_ret[0].shape)
print("\nIntermediate shape of hid[0]:")
print(d1, d2, d3, d4)

print("\n====================== Torch Compile ======================")
_, hid_ret = compiled_lstm(x, h)

print("Return shape of hid[0]:")
print(hid_ret[0].shape)
print("\nIntermediate shape of hid[0]:")
print(d1, d2, d3, d4)