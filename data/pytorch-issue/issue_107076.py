import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo.utils
import torch._dynamo
import logging
torch._dynamo.config.log_level = logging.INFO

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size,  num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

input_size = 784
hidden_size = 500
num_classes = 10

model = SimpleModel(input_size, hidden_size, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim._multi_tensor.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.1)

inputs = torch.randn(64, 784)

labels = torch.randint(0,10,(64,))

num_epochs = 5
def opt_scheduler_step(optimizer):
    optimizer.step()

    return optimizer

opt_scheduler_step = torch.compile(opt_scheduler_step, backend="eager", fullgraph=False)

for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()

    loss.backward()
    optimizer = opt_scheduler_step(optimizer)
    scheduler.step()

    print(loss.item())

class AdamW(torch.optim.AdamW):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        compiled: Optional[bool] = None,
    ):
        if compiled:
            # foreach = False; capturable = False
            # foreach = True; capturable = True
            foreach = False; capturable = True
            lr = torch.tensor(lr, requires_grad=False)
            
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                         amsgrad=amsgrad, maximize=maximize, foreach=foreach, capturable=capturable,
                         differentiable=differentiable, fused=fused)
        if compiled:
            print("*** Using compiled AdamW ***")
            @torch.compile(fullgraph=False)
            def fn(closure=None):
                return self.step_(closure)

            self.step = fn
        else:
            self.step = self.step_

    def step_(self, closure=None):
        return super().step(closure)