import torch

def test_allclose_zero_tolerance(self):
        # Test with zero tolerance
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0 + 1e-9])
        self.assertFalse(torch.allclose(a, b, rtol=0, atol=0))