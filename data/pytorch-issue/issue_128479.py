import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST(nn.Module):
    def __init__(self, num_classes):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        return x

class MnistTest(unittest.TestCase):
    def test_inference(self):
        torch.manual_seed(1 << 30)
        num_classes = 1
        cpu_model = MNIST(num_classes)
        batch_size = 32
        # Inference Test
        test_nums = 2
        cpu_model.eval()
        with torch.no_grad():
            with torch.autograd.profiler.profile() as prof:
                for t in range(test_nums):
                    cpu_images = torch.randn(
                        (batch_size, 1, 28, 28), dtype=torch.float32)
                    # CPU forward
                    cpu_outs = cpu_model(cpu_images)

            print(prof.table(sort_by="count", row_limit=-1))
            prof.export_chrome_trace('./1.json')

if __name__ == '__main__':
    unittest.main()