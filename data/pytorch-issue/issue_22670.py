import torch.nn as nn

import torch
from torch import nn

class SparseTest(nn.Module):
    def __init__(self):
        super(SparseTest, self).__init__()
        self.S = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 2], [2, 3, 0, 3]]),
            values=torch.tensor([1.0, 2.0, 1.0, 3.0]),
            size=[3, 4]).cuda()
        self.fc = nn.Linear(6, 4) 

    def forward(self, x):
        self.S = self.S
        x = torch.spmm(self.S, x)
        x = x.reshape(-1)
        x = self.fc(x)
        return x

if __name__ == "__main__":

    X = torch.ones(4, 2, dtype=torch.float).cuda()
    y = torch.zeros(4, dtype=torch.float).cuda()
    sparseTest = SparseTest()
    sparseTest = sparseTest.cuda()
    sparseTest = torch.nn.DataParallel(sparseTest)  # whether use DataParallel
    optimizer = torch.optim.Adam(sparseTest.parameters(), lr=0.001, weight_decay=0.00005)
    lossMSE = nn.MSELoss()
    with torch.set_grad_enabled(True):
        for i in range(10):
            x = sparseTest(X)
            optimizer.zero_grad()
            loss = lossMSE(x, y)
            loss.backward()
            optimizer.step()
            print("loss: {:.8f}".format(loss.item()))

def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()   # N x Mp x Fin
        return x

def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        x = x.permute(1, 2, 0).contiguous()  #M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()   # N x Mp x Fin
        return x