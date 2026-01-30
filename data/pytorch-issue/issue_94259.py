import torch

self.assertEqual(torch.zeros_like(cpu_out), torch.zeros_like(mps_out))