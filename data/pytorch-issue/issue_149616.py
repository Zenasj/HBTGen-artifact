import torch

if self.key_cache[layer_idx].device.type == "meta":
            self.key_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx], device=key_states.device)
            self.value_cache[layer_idx] = torch.zeros_like(self.value_cache[layer_idx], device=value_states.device)