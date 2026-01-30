import torch
class TestClass:
    def test_grid_sampler_2d(self):
        torch.manual_seed(0)
        b = torch.rand(2, 13, 10, 2, dtype=torch.float64)
        a = torch.rand(2, 3, 5, 20, dtype=torch.float64)
        torch.grid_sampler_2d(a, b, interpolation_mode=0, padding_mode=0, align_corners=False)