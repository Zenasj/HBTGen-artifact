import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding_size = 16
        self.filter_num = 512
        self.padding_length = 25
        self.convolutions = nn.ModuleList([nn.Conv1d(1, self.filter_num // 8, kernel_size=(K, self.embedding_size), stride=1) for K in range(1, 9)])

    def forward(self):
        X = torch.randn([300, 1, self.padding_length, self.embedding_size])
        X = [torch.tanh(convolution(X).squeeze(3)) for convolution in self.convolutions]
        X = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in X]
        X = torch.cat(X, dim=1)
        return X

if __name__ == "__main__":
    model = Model()
    output = model()
    output.mean().backward()

conv1d = nn.Conv1d(1, 32, kernel_size=(8, 16), stride=1)
x = torch.randn(300, 1, 25, 16)
x.requires_grad_()
y = conv1d(x).sum()
y.backward()