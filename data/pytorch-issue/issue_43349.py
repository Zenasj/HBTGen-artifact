py
import torch
mag, phase = torch.tensor(5., requires_grad=True), torch.tensor(3., requires_grad=True)

complex_good = torch.view_as_complex(torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1))
complex_good.backward()  # works

complex_bad = mag * torch.exp(1j * phase)
complex_bad.backward()