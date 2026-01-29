import torch
import torch.nn as nn

class SpatialAttn(nn.Module):
    def __init__(self, in_plane):
        super(SpatialAttn, self).__init__()
        self.cal_Q = nn.Conv2d(in_plane, in_plane // 8, 1)
        self.cal_K = nn.Conv2d(in_plane, in_plane // 8, 1)
        self.softmax = nn.Softmax(dim=1)
        self.cal_V = nn.Conv2d(in_plane, in_plane, 1)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        print('x.shape = ', x.shape)
        Q = self.cal_Q(x)
        K = self.cal_K(x)
        QQ = Q.view(Q.shape[0], Q.shape[1], Q.shape[2] * Q.shape[3])
        KK = K.view(K.shape[0], K.shape[1], K.shape[2] * K.shape[3])
        A = self.softmax(torch.bmm(torch.transpose(QQ, 1, 2), KK))
        V = self.cal_V(x)
        VV = V.view(V.shape[0], V.shape[1], V.shape[2] * V.shape[3])
        Xs = self.alpha * torch.bmm(VV, A) + x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        XS = Xs.view(Xs.shape[0], Xs.shape[1], x.shape[2], x.shape[3])
        return XS

# torch.rand(B, 3, 32, 32, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.spatial_attn = SpatialAttn(64)
        self.fc = nn.Linear(64 * 32 * 32, 10)  # Assumes 32x32 spatial dimensions after attention

    def forward(self, x):
        x = self.conv1(x)
        x = self.spatial_attn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Arbitrary batch size
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

