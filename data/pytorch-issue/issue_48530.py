# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dummy convolution layer to simulate feature extraction
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # Matches GetInput() output channels
        
    def project_tensorflow(self, x, y, img_size, img_feat):
        # Original code with index order bug (x, y instead of y, x)
        x = torch.clamp(x, min=0, max=img_size[1] - 1)
        y = torch.clamp(y, min=0, max=img_size[0] - 1)
        
        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()
        
        # ERROR: Indices are in (x, y) order instead of (y, x)
        Q11 = img_feat[:, x1, y1].clone()  # Should be img_feat[:, y1, x1]
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()
        
        # Simplified return to trigger error
        return Q11
    
    def forward(self, x):
        img_feat = self.conv(x)
        img_size = (img_feat.size(2), img_feat.size(3))  # (H, W)
        # Coordinates designed to trigger out-of-bounds (W exceeds H dimension)
        x_coord = torch.tensor([[4.0]])  # W=5 (max 4), H=3 (max 2)
        y_coord = torch.tensor([[2.0]])
        return self.project_tensorflow(x_coord, y_coord, img_size, img_feat)

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B=1, C=3, H=3, W=5) to trigger index error when W exceeds H dimension
    return torch.rand(1, 3, 3, 5, dtype=torch.float32)

