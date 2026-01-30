import torch
import torch.nn as nn
import torch.nn.functional as F

import tempfile
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def save_load_ge(ge):
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        f.close()
        ge.save(f.name)
        imported = torch.jit.load(f.name)
    finally:
        os.unlink(f.name)
    return imported


device = 'cuda'
module = Net().to(device).eval()
inputs_cuda = [torch.rand(5, 1, 28, 28, device='cuda')]
inputs_cpu = [torch.rand(5, 1, 28, 28, device='cpu')]
ge = torch.jit.trace(module, inputs)

# Test .cpu(), .cuda() on ge
print('Test .cpu(), .cuda() on ScriptModule')
ge_cpu = ge.cpu()
ge_cpu(*inputs_cpu)
ge_cuda = ge_cpu.cuda()
ge_cuda(*inputs_cuda)

# Test .cpu(), .cuda() on ge after export/import
print('Test .cpu(), .cuda() after export/import')
ge = save_load_ge(ge)
ge_cuda = ge.cuda()
ge_cuda(*inputs_cuda)  # FAIL, still expecting CPU tensor