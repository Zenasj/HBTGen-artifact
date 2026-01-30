import torch
torch.hub.set_dir("/opt/models/xlmr.large")
model = torch.hub.load('pytorch/fairseq', 'xlmr.large')