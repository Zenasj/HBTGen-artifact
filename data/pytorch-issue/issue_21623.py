# torch.rand(1, dtype=torch.float32)  # Dummy input
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Dummy parameter for optimizer
        # Scenario 1: epoch fixed at 2 (bug scenario)
        self.opt1 = optim.SGD([self.linear.weight], lr=1.0)
        self.sched1 = StepLR(self.opt1, step_size=2, gamma=2.0)
        # Scenario 2: epoch fixed at 3 (non-bug scenario)
        self.opt2 = optim.SGD([self.linear.bias], lr=1.0)
        self.sched2 = StepLR(self.opt2, step_size=2, gamma=2.0)
        
        # Determine if bug exists during initialization
        self.bug_exists = self._run_and_compare()
    
    def _run_and_compare(self):
        # Run scenario 1 (epoch=2)
        for _ in range(5):
            for _ in range(2):
                self.opt1.step()
            self.sched1.step(epoch=2)
        lr1 = self.sched1.get_last_lr()[0]
        
        # Run scenario 2 (epoch=3)
        for _ in range(5):
            for _ in range(2):
                self.opt2.step()
            self.sched2.step(epoch=3)
        lr2 = self.sched2.get_last_lr()[0]
        
        # Bug exists if scenario1 LR is 32 (2^5) and scenario2 stays at 1
        return lr1 == 32.0 and lr2 == 1.0
    
    def forward(self, x):
        # Return bug existence flag as tensor
        return torch.tensor([float(self.bug_exists)], dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

