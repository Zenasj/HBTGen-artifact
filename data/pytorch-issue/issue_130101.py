# torch.rand(1, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('document_id', torch.zeros(32768, dtype=torch.int, device='cuda'))
        self.document_id[:4096] = 0
        for i in range(4096, 32768, 2048):
            self.document_id[i:i+2048] = i // 2048
        self.B = 4
        self.H = 1
        self.M = 32768
        self.N = 32768
        self.KV_BLOCK_SIZE = 128
        self.Q_BLOCK_SIZE = 128

    @staticmethod
    def broadcast_to_dim(x, dim):
        while x.dim() < dim:
            x = x.unsqueeze(0)
        return x

    def _convert_mask_to_block_mask(self, mask):
        assert mask.dtype == torch.bool
        mask = self.broadcast_to_dim(mask, 4)
        B, H, Q, KV = mask.shape
        assert Q % self.Q_BLOCK_SIZE == 0
        assert KV % self.KV_BLOCK_SIZE == 0
        mask = mask.view(
            B, H, Q // self.Q_BLOCK_SIZE, self.Q_BLOCK_SIZE,
            KV // self.KV_BLOCK_SIZE, self.KV_BLOCK_SIZE
        )
        mask = mask.permute(0, 1, 2, 4, 3, 5)
        mask = mask.sum(dim=[-2, -1]) > 0
        return mask

    def _create_block_mask_from_mask(self, block_mask):
        device = block_mask.device
        block_mask = block_mask.to(dtype=torch.int8)
        kv_num_blocks = block_mask.sum(dim=3)
        kv_indices = torch.argsort(
            block_mask, dim=3, descending=True, stable=True
        )
        q_num_blocks = block_mask.sum(dim=2)
        q_indices = torch.argsort(
            block_mask, dim=2, descending=True, stable=True
        ).permute(0, 1, 3, 2)
        return (
            kv_num_blocks.to(torch.int32).contiguous(),
            kv_indices.to(torch.int32).contiguous(),
            q_num_blocks.to(torch.int32).contiguous(),
            q_indices.to(torch.int32).contiguous(),
            torch.tensor([self.KV_BLOCK_SIZE], dtype=torch.int32, device=device),
            torch.tensor([self.Q_BLOCK_SIZE], dtype=torch.int32, device=device),
        )

    def forward(self, x):
        device = x.device
        qk = torch.zeros(1, 1, self.M, self.N, device=device)
        q_idx = torch.arange(self.M, device=device).view(1, 1, self.M, 1)
        kv_idx = torch.arange(self.N, device=device).view(1, 1, 1, self.N)
        causal_mask = q_idx <= kv_idx
        document_mask = self.document_id[q_idx] == self.document_id[kv_idx]
        combined_mask = causal_mask & document_mask
        qk_masked = torch.where(combined_mask, qk, -float('inf'))
        mask = torch.isneginf(qk_masked)
        mask = ~mask  # mask is True where qk_masked is not -inf

        block_mask = self._convert_mask_to_block_mask(mask)
        return self._create_block_mask_from_mask(block_mask)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, device='cuda')

