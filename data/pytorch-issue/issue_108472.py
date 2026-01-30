import torch

def test_adding_tensor_offsets(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x[16:32]

        with torch.no_grad():
            x = torch.randn(1024, device=self.device)
            self.assertEqual(fn(x[0:]), x[16:][:16])
            self.assertEqual(fn(x[128:]), x[128 + 16 :][:16])