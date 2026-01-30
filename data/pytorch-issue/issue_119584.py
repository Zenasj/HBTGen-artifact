import torch

from torch._subclasses import fake_tensor
import transformers

fake_mode = fake_tensor.FakeTensorMode(allow_non_fake_inputs=False)
with fake_mode:
    fake_model = transformers.AutoModel.from_pretrained("sshleifer/tiny-gpt2")