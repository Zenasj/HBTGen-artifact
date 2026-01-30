import torch

with self.assertRaisesRegex(RuntimeError, "The algorithm failed to converge"):
    svd(a)

a = torch.eye(3, 3, dtype=torch.float32, device='cpu')
torch.svd(a)