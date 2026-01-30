import torch

def test_floordiv(self):
        def fn(a, i):
            n = (i * 1.0) // 8.0
            return a + n

        self.common(fn, (make_tensor(100, device="cpu", dtype=torch.float32), 12))