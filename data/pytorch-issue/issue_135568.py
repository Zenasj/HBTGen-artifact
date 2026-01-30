import torch
import torch._dynamo as dynamo
import torch._inductor as inductor
import torch.nn as nn

class Level1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = Level2()
        self.linear3 = Level3()

    def forward(self, x1):
        torch._dynamo.mark_dynamic(x1, 0)   #(3) Error Attempt to trace forbiddden call error - (2) & (3)
        x = self.linear1(x1)
        x = self.linear2(x)
        return self.linear3(x)

class Level2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x2):
        x2 = self.linear(x2)
        return x2
    
class Level3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.non_linear = torch.nn.ReLU()

    def forward(self, x3):
        x3 = self.linear(x3)
        return self.non_linear(x3)

x = torch.zeros((3, 1))
# torch._dynamo.mark_dynamic(x, 0) #(1) Solution  - It's functional

model = Level1()
model = torch.compile(model,backend="inductor", dynamic=None)

# Vary the batch size (1st dimension) of input
for bs in range(2, 10):
    x = torch.zeros((bs, 1))
    torch._dynamo.mark_dynamic(x, 0)  #(2) Error Constraint Violation error - Only (2)
    model(x)
print(f"Test Completed....")

import torch
import torch._dynamo as dynamo
import torch._inductor as inductor
import torch.nn as nn
 
class Level1(torch.nn.Module):
 
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 3)
        self.linear2 = Level2()
        self.linear3 = Level3()
 
    def forward(self, x1):
        #torch._dynamo.mark_dynamic(x1, 0)   #(3) Error Attempt to trace forbiddden call error - (2) & (3)
        x = self.linear1(x1)
        x = self.linear2(x)
        return self.linear3(x)
 
class Level2(torch.nn.Module):
 
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x2):
        x2 = self.linear(x2)
        return x2
   
class Level3(torch.nn.Module):
 
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.non_linear = torch.nn.ReLU()
 
    def forward(self, x3):
        x3 = self.linear(x3)
        return self.non_linear(x3)
 
x = torch.zeros((3, 1))
torch._dynamo.mark_dynamic(x, 0) #(1) Solution  -Throws Assertin error
 
model = Level1()
model = torch.compile(model,backend="inductor", dynamic=None)
 
model(x)

import torch
import torch._dynamo as dynamo
import torch._inductor as inductor
import torch.nn as nn

class Level1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = Level2()
        self.linear3 = Level3()

    def forward(self, x1):
        # torch._dynamo.mark_dynamic(x1, 0)   #(3) Error Attempt to trace forbiddden call error - (2) & (3)
        x = self.linear1(x1)
        x = self.linear2(x)
        return self.linear3(x)

class Level2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x2):
        x2 = self.linear(x2)
        return x2
    
class Level3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.non_linear = torch.nn.ReLU()

    def forward(self, x3):
        x3 = self.linear(x3)
        return self.non_linear(x3)

x = torch.zeros((3, 2))
# torch._dynamo.mark_dynamic(x, 0) #(1) Solution  - It's functional

model = Level1()
model = torch.compile(model,backend="inductor", dynamic=None)

# Vary the batch size (1st dimension) of input
for bs in range(2, 10):
    x = torch.zeros((bs, 2))
    torch._dynamo.mark_dynamic(x, 0)  #(2) Error Constraint Violation error - Only (2)
    model(x)
print(f"Test Completed....")

import torch
import torch._dynamo as dynamo
import torch._inductor as inductor
import torch.nn as nn

class Level1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = Level2()
        self.linear3 = Level3()

    def forward(self, x1):
        x = self.linear1(x1)
        x = self.linear2(x)
        return self.linear3(x)

class Level2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x2):
        x2 = self.linear(x2)
        return x2
    
class Level3(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.non_linear = torch.nn.ReLU()

    def forward(self, x3):
        x3 = self.linear(x3)
        return self.non_linear(x3)

model = Level1()
model = torch.compile(model,backend="inductor", dynamic=None)

# Vary the batch size (1st dimension) of input
for bs in range(2, 10):
    x = torch.zeros((bs, 2))
    #torch._dynamo.mark_dynamic(x, 0, min=2, max=12)  #(2) Error Constraint Violation error
    torch._dynamo.decorators.mark_unbacked(x, 0)
    model(x)
print(f"Test Completed....")