import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, dropout=True):
        super(ConvBlock, self).__init__()

        self.dropout = dropout
        self.conv = torch.nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=ks)
        self.nl = nn.PReLU()

    def forward(self, input):

        output = self.conv(input)
        output = self.nl(output)
        if self.dropout:
            output = F.dropout(output)
        return output

for tag, value in model.named_parameters():
    if value.grad is not None :
        print("{} : {} : {}".format(tag, torch.norm(value.grad), torch.std(value.grad)))