import torch
import torch.nn as nn

def test_conv1d_circular_padding(self):
        y_cpu = torch.randn(32, 8, 64)
        conv_cpu = nn.Conv1d(8, 128, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        conv_gpu = copy.deepcopy(conv_cpu).to(device='mps')
        x_cpu = conv_cpu(y_cpu)

        y_gpu = y_cpu.to(device='mps')
        x_gpu = conv_gpu(y_gpu)
        self.assertEqual(x_cpu, x_gpu.cpu())