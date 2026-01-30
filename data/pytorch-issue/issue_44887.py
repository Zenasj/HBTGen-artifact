import torch

def forward(self, x1, x2):
        assert x1.shape == x2.shape
        d = self.d
        s1 = self.s1
        s2 = self.s2
        n, c, h, w = x1.shape
        out_h = (h - 1) // s1 + 1
        out_w = (w - 1) // s1 + 1
        out_k = 2 * d // s2 + 1
        result = torch.zeros(n, out_k ** 2, out_h, out_w, device=x1.device)
        ...