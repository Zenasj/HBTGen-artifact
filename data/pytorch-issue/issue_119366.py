import torch
optimizer = torch.optim.AdamW(model.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler.get_last_lr()