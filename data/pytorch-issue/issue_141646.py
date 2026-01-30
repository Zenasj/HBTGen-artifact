import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 10) 

    def forward(self, x):
        x = self.fc1(x)
        return x

compile_kwargs = {"dynamic": False}
device = torch.device('cpu')

model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_engine, optimizer, *_ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    optimizer=optimizer,
    config="./deepspeed_config.json",
)
# torch_compile
model_engine.compile(
    compile_kwargs=compile_kwargs
)

for epoch in range(100):
    with torch._dynamo.compiled_autograd.enable(
                torch.compile(backend=get_accelerator().get_compile_backend(), **compile_kwargs)):
        inputs=torch.randn([8,32], requires_grad=True)
        labels=torch.ones([8])
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model_engine(inputs)
        loss = criterion(outputs.float(), labels.long())
        model_engine.backward(loss)
        model_engine.step()
print("Finished Training")