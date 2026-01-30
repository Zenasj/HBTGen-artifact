import torch

url = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
model_weights = torch.hub.load_state_dict_from_url(url)