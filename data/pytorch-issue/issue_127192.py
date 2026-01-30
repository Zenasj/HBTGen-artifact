import torch.nn as nn

import sklearn.model_selection

import torch
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = dim

        self.l1 = torch.nn.Linear(dim, 300, bias=False)
        self.b1 = torch.nn.BatchNorm1d(300)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = self.l1(x)
        h = self.b1(h)
        return h
    
if __name__ == '__main__':
    model = MultiLayerPerceptron(5)
    
    import torch
    X = torch.randn([1,300, 5]).to('cpu')
    print(model(X))