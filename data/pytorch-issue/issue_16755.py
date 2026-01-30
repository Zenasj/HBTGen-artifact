import torch.nn as nn
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)
        
    @property
    def l2(self):
        return nn.Linear(2,3)

test_module = TestModule()

list(test_module.children())

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1,2)
        self.l2 = nn.Linear(1,2)

class MyModule(TestModule):
    def __init__(self):
        super().__init__()
        self.l2 = nn.Linear(1, 10)