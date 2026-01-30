import torch
N = 256
x = torch.randn(N * N, 2, 2).cuda()
y = torch.inverse(x)
torch.matmul(y, x)  # this is not a batch of identity matrices, in fact y == x

def batchedInv(self, batchedTensor):
        if batchedTensor.shape[0] >= 256 * 256 - 1:
            temp = []
            for t in torch.split(batchedTensor, 256 * 256 - 1):
                temp.append(torch.inverse(t))
            return torch.cat(temp)
        else:
            return torch.inverse(batchedTensor)