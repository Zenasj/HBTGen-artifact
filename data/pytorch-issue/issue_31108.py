import torch

# The following ones will fail.
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int64).pow(4); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int32).pow(4); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int16).pow(4); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int8).pow(4); print(a)

# The following ones will be okay.
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int64).pow(2); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int32).pow(2); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int16).pow(2); print(a)
a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int8).pow(2); print(a)

def test_long_tensor_pow_floats(self, device):
        ints = [0, 1, 23, 4567] # int64_t
        floats = [0.0, 1 / 3, 1 / 2, 1.0, 3 / 2, 2.0] #float
        tensor = torch.tensor(ints, dtype=torch.int64, device=device)
        for pow in floats:
            self._test_pow(tensor, pow)