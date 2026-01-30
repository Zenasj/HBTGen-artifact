import torch

def check_nan(tensor):
    return torch.isnan(tensor).any().item()

loss_is_nan = torch.tensor(check_nan(loss), device=accelerator.device, dtype=torch.int)

# reduce to check if any process has NaN loss
loss_is_nan = accelerator.reduce(loss_is_nan, reduction="sum")

if loss_is_nan.item() == 0:
    accelerator.backward(loss)
    # ...
else:
    if accelerator.is_main_process:
        print("Loss has NaN, skip...")