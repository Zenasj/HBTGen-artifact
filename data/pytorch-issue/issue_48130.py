import torch

# device = 'cuda'
@dtypes(*torch.testing.get_all_int_dtypes())
def test_fmod_by_zero_integral(self, device, dtype):
    x = make_tensor((10, 10), device=device, dtype=dtype, low=-9, high=9)
    zero = torch.zeros_like(x)
    if dtype == torch.int64:
        self.assertEqual(x.fmod(zero) == 4294967295, x >= 0)
        self.assertEqual(x.fmod(zero) == -1, x < 0)
    else: # torch.uint8, torch.int8, torch.int16, torch.int32
        value = 255 if dtype == torch.uint8 else -1
        self.assertTrue(torch.all(x.fmod(zero) == value))