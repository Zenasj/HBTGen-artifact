import torch
import time
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
import torch.nn as nn
import time
import torchvision
import torch.utils._benchmark as benchmark_utils

device = "cuda"
model = torchvision.models.resnet.resnet101(pretrained=True).to(device)
targets = torch.randint(0, 1000, (100, 100), device=device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=1e-3) # <----------------------- optimizer. 
                                                          # would compare optim.SGD vs optim._multi_tensor.SGD
running_loss = 0.0
target = torch.empty(128, dtype=torch.long, device=device).random_(5)

optimizer.zero_grad()
inputs = torch.rand(128, 3, 100, 100, device=device , requires_grad=True)
outputs = model(inputs)
loss = criterion(outputs, target) 
loss.backward()
optimizer.step()
running_loss += loss.item()

def main():
    timer = benchmark_utils.Timer(
        stmt="optimizer.step()",
        globals=globals(),
        label="str(optimizer)",
    )

    for i in range(1):
        print(f"Run: {i}\n{'-' * 40}")
        print(f"timeit:\n{timer.timeit(1000)}\n")
        print(f"autorange:\n{timer.blocked_autorange()}\n\n")


if __name__ == "__main__":
    main()