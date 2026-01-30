import torch
import torch.nn as nn

def test_nonstrict_sequential_slicing(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.seq_last = self.seq[1:]
                return self.seq_last(x)

        ep = export(TestModule(), (torch.randn(4, 4),), strict=False)

def test_sequential_slicing(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                seq_last = self.seq[1:] # instead of assigning to self.seq_last, which strict dislikes
                return seq_last(x)

        ep = export(TestModule(), (torch.randn(4, 4),))