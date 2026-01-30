import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self, output_padding):
        super().__init__()
        
        self.c1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                                     padding=1, output_padding=output_padding)
        self.c2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                                     padding=1, output_padding=output_padding)
        self.c3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2,
                                     padding=1, output_padding=output_padding)
    def forward(self, x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = self.c2(x)
        x = nn.functional.relu(x)
        x = self.c3(x)
        x = nn.functional.relu(x)
        return x

def test(output_padding):
    network = Test(output_padding=output_padding).cuda()
    x = torch.rand((1028, 128, 6, 6)).cuda()


    for i in range(5):
        start = time.time()
        for i in range(10):
            network(x)
        print((time.time() - start) / 10)

test((0, 0))
test((1, 1))