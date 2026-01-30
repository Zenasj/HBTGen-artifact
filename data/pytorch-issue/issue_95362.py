import torch

w = {k: v for k, v in model.state_dict().items()}
torch.save(w, Folder + 'ScriptUntrained_LibTorchTest03Cmb.ptsd')