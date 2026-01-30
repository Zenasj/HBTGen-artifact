import os
import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x.view(x.size(0), -1))
        return out


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_cuda = torch.cuda.is_available() # you can change here with False

    cla_func = Classifier(input_dim=512, output_dim=10)
    print(cla_func)
    if use_cuda:
        cla_func.cuda()

    with torch.no_grad():
        x = torch.rand([16, 512]).float()
        w = torch.rand([16, 10]).float()
        xw = torch.cat([x, w], dim=1)

        if use_cuda:
            xw = xw.cuda()

        print(xw.shape)
        y = cla_func(xw)
        print(y.shape)