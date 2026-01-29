# torch.randint(0, 8192, (B, H, W), dtype=torch.long)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed_weight = nn.Parameter(torch.randn(8192, 32))
        self.patch_embed_proj = nn.Linear(32, 768)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))

    def random_masking(self, x, len_keep):
        N, L, D = x.shape
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones([N, L], dtype=torch.bool, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, ids_restore

    def forward_features(self, x):
        # Embedding layer
        embedding = F.embedding(x, self.patch_embed_weight)
        # Projection to 768 dimensions
        proj = self.patch_embed_proj(embedding)
        B, H, W, D = proj.shape
        # Reshape to (B, H*W, D)
        x = proj.view(B, H * W, D)
        # Compute masking parameters
        L = x.shape[1]
        len_keep = int(L * (1 - self.mask_ratio))
        # Apply masking
        mask, _ = self.random_masking(x, len_keep)
        # Clone to avoid view modification error
        x = x.clone()
        x[mask] = self.mask_token.to(x.dtype)
        return x, mask

    def forward(self, x):
        return self.forward_features(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, H, W = 1, 14, 14
    return torch.randint(0, 8192, (B, H, W), dtype=torch.long)

