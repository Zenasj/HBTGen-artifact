# torch.rand(B, C, H, W, dtype=torch.float32), torch.randint(0, C, (B, H, W))
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels, num_classes, device, dtype):
    # Minimal one-hot implementation for compatibility
    batch_size, H, W = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, H, W, device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1)

class MyModel(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none'):
        super(MyModel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma  # Stored as float to avoid device mismatches
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, x):
        input, target = x
        if not torch.is_tensor(input):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")
        if not len(input.shape) == 4:
            raise ValueError(f"Invalid input shape, expected BxNxHxW. Got {input.shape}")
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("Input and target spatial dimensions do not match")
        if input.device != target.device:
            raise ValueError(f"Input and target devices mismatch: {input.device} vs {target.device}")

        input_soft = F.softmax(input, dim=1) + self.eps
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)
        
        gamma_tensor = torch.tensor(self.gamma, device=input.device, dtype=input.dtype)
        weight = torch.pow(1.0 - input_soft, gamma_tensor)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss

def my_model_function():
    return MyModel(alpha=0.5, gamma=2.0, reduction='mean')

def GetInput():
    B, C, H, W = 2, 3, 4, 4
    input = torch.rand(B, C, H, W, dtype=torch.float32, device='cuda')
    target = torch.randint(0, C, (B, H, W), device='cuda')
    return (input, target)

