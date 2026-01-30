import torch

with torch.autocast(device_type='cuda', dtype=torch.float32):
            gru_seq, gru_last = self.gru(emb, hidden)