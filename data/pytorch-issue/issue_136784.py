import torch
import torch.nn as nn

def test_rrelu_noise_mutation(self):
        def fn(x):
            noise = torch.ones_like(x)
            result = torch._C._nn.rrelu_with_noise(x, noise, 0.2, 0.8, True)
            return result, noise

        x = -torch.abs(torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True))

        ref_y, ref_noise = fn(x)
        self.assertTrue(torch.all(ref_noise < torch.ones_like(ref_noise)).item())

        comp_y, comp_noise = torch.compile(fn, backend="inductor", fullgraph=True)(x)
        self.assertTrue(torch.all(comp_noise < torch.ones_like(comp_noise)).item())