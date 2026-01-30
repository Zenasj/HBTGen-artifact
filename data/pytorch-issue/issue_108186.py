import torch.nn as nn

from torch._subclasses import fake_tensor
import transformers

fake_mode = fake_tensor.FakeTensorMode()
with fake_mode:
    fake_model = transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2")

import tempfile
import torch
from torch._subclasses import fake_tensor

class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.fc1 = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.fc1(x)

with tempfile.NamedTemporaryFile() as state_dict_file:
    # Create state_dict to be loaded later
    model = TheModelClass()
    torch.save(model.state_dict(), state_dict_file.name)

    fake_mode = fake_tensor.FakeTensorMode()
    with fake_mode:
        # This is where the bug is triggered
        state_dict = torch.load(state_dict_file.name)

from torch._subclasses import fake_tensor
import transformers

fake_mode = fake_tensor.FakeTensorMode()
with fake_mode:
    fake_model = transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2")  # raises OSError: Unable to load weights from pytorch checkpoint file for '...' at  If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True.