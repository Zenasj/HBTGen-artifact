import torch

map_location

_use_new_zipfile_serialization

os.environ['CUDA_VISIBLE_DEVICES']='3'
torch.cuda.set_device(3)

torch.save(model_.state_dict(), 'model_best_bacc.pth.tar', _use_new_zipfile_serialization=False)

torch.load('model_best_bacc.pth.tar',map_location='cpu')