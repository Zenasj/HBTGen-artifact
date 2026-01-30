import torch.nn as nn

import torch                                                                                                                                                                                                                                                                                                                                                                                                                      

class MyModel(torch.nn.Module):
    def __init__(self, input_dim=16, output_dim=32):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, output_dim, 3, 1, padding="same")
        self.out = torch.nn.Linear(input_dim * output_dim, output_dim)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Process
        # x = x.unsqueeze(1) # (batch_size, 1, seq_len, input_dim)
        x = self.conv(x) # (batch_size, channels, seq_len, input_dim)
        b, c, t, f = x.size()
        # x = self.out(x.transpose(1, 2).reshape(b, t, c * f)) # (batch_size, seq_len, output_dim)
        x = self.out(x.reshape(b, t, c * f))
        # Permute for loss
        # logits = x.permute((0, 2, 1)) # (batch_size, output_dim, seq_len)
        logits = x.reshape(x.size(0), x.size(2), x.size(1))
        # Breaks during backward
        loss = torch.nn.functional.cross_entropy(logits, targets)

        return loss


# Parameters
device = "cuda"
batch_size = 48
seq_len = 144
input_dim = 39 # feature_dim
num_classes = 111

# Model init
model = MyModel(input_dim, num_classes)
model.to(device)
model = torch.compile(model, fullgraph=True, mode="default")

# Data
x = torch.ones((batch_size, 1, seq_len, input_dim), device=device)
targets = torch.randint(
0, num_classes - 1, (batch_size, seq_len), device=device, dtype=torch.int64
)

# Results in: "RuntimeError: grad_input must be contiguous"
loss = model(x, targets)
loss.backward()