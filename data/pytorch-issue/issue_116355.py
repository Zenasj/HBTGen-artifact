import torch

# When saving the checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, filename, _use_new_zipfile_serialization=False)

# When loading the checkpoint
chkpoint = torch.load(filename, map_location=torch.device('cpu'))