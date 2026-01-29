# torch.rand(1, dtype=torch.int)  # Example input: single integer tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Buggy version uses local coordinator_rank (0), Fixed uses global (1)
        self.buggy = BuggyDistWrapper(coordinator_rank=0)
        self.fixed = FixedDistWrapper(coordinator_rank=1)

    def forward(self, obj):
        # Compare dst values used in gather_object between versions
        buggy_dst = self.buggy.gather_object(obj)
        fixed_dst = self.fixed.gather_object(obj)
        # Return 1 if destinations differ, 0 otherwise (as tensor)
        return torch.tensor(1 if buggy_dst != fixed_dst else 0)

class BuggyDistWrapper(nn.Module):
    def __init__(self, coordinator_rank):
        super().__init__()
        self.coordinator_rank = coordinator_rank  # Local rank (buggy)

    def gather_object(self, obj):
        # Returns the dst value used (local coordinator_rank)
        return self.coordinator_rank

class FixedDistWrapper(nn.Module):
    def __init__(self, coordinator_rank):
        super().__init__()
        self.coordinator_rank = coordinator_rank  # Global rank (fixed)

    def gather_object(self, obj):
        # Returns the dst value used (global coordinator_rank)
        return self.coordinator_rank

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random integer tensor (mock object for gather)
    return torch.randint(0, 10, (1,), dtype=torch.int)

