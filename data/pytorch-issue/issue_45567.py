import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FakeAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_w = nn.Parameter(torch.Tensor(3, 5))
        self.dec_w = self.enc_w.T

        nn.init.normal_(self.enc_w)

    def encode(self, data):
        return F.linear(data, self.enc_w)

    def decode(self, data):
        # return F.linear(data, self.enc_w.T)  # workaround the error
        return F.linear(data, self.dec_w)

    def forward(self, data):
        return self.decode(self.encode(data))


def main():
    device = 'cuda'
    # device = 'cpu'  # works!
    ae = FakeAE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ae.parameters(), lr=1)

    data = torch.rand((2, 5)).to(device)
    rec = ae(data)
    loss = criterion(data, rec)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()