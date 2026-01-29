import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, 9, 1024, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self, num_points=1024):
        super(MyModel, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(9, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 128, 1)
        self.conv10 = nn.Conv1d(128, 2, 1)
        self.max_pool = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        point_features = out
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        global_feature = self.max_pool(out)
        global_feature_repeated = global_feature.repeat(1, 1, self.num_points)
        out = torch.cat([global_feature_repeated, point_features], 1)
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = F.relu(self.conv8(out))
        out = F.relu(self.conv9(out))
        out = self.conv10(out)
        out = out.transpose(2, 1).contiguous()
        out = F.log_softmax(out.view(-1, 2), dim=1)
        out = out.view(batch_size, self.num_points, 2)
        return out

def my_model_function():
    # Return model instance initialized on CUDA
    return MyModel().cuda()

def GetInput():
    # Generate random input matching (batch_size, channels=9, num_points=1024)
    B = 32  # Batch size from original issue's code
    return torch.randn(B, 9, 1024, dtype=torch.float32).cuda()

