import torch
import torch.nn as nn

def remove_all_spectral_norm(item):
    if isinstance(item, nn.Module):
        try:
            nn.utils.remove_spectral_norm(item)
        except Exception:
            pass
        
        for child in item.children():  
            remove_all_spectral_norm(child)

    if isinstance(item, nn.ModuleList):
        for module in item:
            remove_all_spectral_norm(module)

    if isinstance(item, nn.Sequential):
        modules = item.children()
        for module in modules:
            remove_all_spectral_norm(module)

for layer in model.modules():
  torch.nn.utils.remove_spectral_norm(layer)

raise ValueError("spectral_norm of '{}' not found in {}".format(
            name, module))