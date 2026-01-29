# torch.rand(4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class CorrectAssign(nn.Module):
    def forward(self, x):
        x_copy = x.clone()
        x_copy[[1, 2]] = 1.0
        return x_copy

class IncorrectAssign(nn.Module):
    def forward(self, x):
        x_copy = x.clone()
        x_copy[1, 2] = 1.0
        return x_copy

class CorrectSelect(nn.Module):
    def forward(self, x):
        return x[[1, 2]]

class IncorrectSelect(nn.Module):
    def forward(self, x):
        return x[1, 2]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.correct_assign = CorrectAssign()
        self.incorrect_assign = IncorrectAssign()
        self.correct_select = CorrectSelect()
        self.incorrect_select = IncorrectSelect()

    def forward(self, x):
        # Assignment test
        x_assign = x.clone()
        correct_a = self.correct_assign(x_assign)
        incorrect_a = self.incorrect_assign(x_assign)
        diff_a = torch.any(correct_a != incorrect_a)

        # Selection test
        correct_s = self.correct_select(x)
        incorrect_s = self.incorrect_select(x)
        diff_s = torch.tensor(correct_s.shape != incorrect_s.shape, dtype=torch.bool)

        return diff_a | diff_s

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, dtype=torch.float32)

