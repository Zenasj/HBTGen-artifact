import torch
import torch.nn as nn
from torch.nn import functional as F

def concat_all_gather(tensor):
    """Stub for DDP's all_gather. Returns input as-is for single-process execution."""
    return tensor

# torch.rand(B, C, H, W, dtype=torch.float32) ← inferred input shape (e.g., 2x3x224x224)
class MyModel(nn.Module):
    def __init__(self, dim=128, K=65536, m=0.999, T=0.07):
        super(MyModel, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Encoder architecture (simplified for demonstration)
        self.encoder_q = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, dim)
        )
        self.encoder_k = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, dim)
        )
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue and pointer initialization
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + (1 - self.m) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = self.queue_ptr[0]
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # Tensor-based arithmetic to avoid SymInt issues
        self.queue_ptr[0] = ptr

    def forward(self, x):
        q = F.normalize(self.encoder_q(x), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(x), dim=1)
        self._dequeue_and_enqueue(k)
        return q, k  # Simplified output for minimal repro

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

