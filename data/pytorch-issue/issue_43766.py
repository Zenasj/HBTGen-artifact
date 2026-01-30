is_zipfile

import torch
state_dict = torch.load('model_weight.pt', map_location="cpu")
torch.save(state_dict, 'model_weight.pt', _use_new_zipfile_serialization=False)