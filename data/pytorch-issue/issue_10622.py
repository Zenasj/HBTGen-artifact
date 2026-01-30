import torch

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'
    
checkpoint = torch.load(load_path, map_location=map_location)