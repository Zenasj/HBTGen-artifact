import torch

array = torch.zeros(1000000)
slice = array[:1000]
clone = slice.clone()

torch.save(array, 'array.pt') # 3.9MB on disk
torch.save(slice, 'slice.pt') # 3.9MB on disk
torch.save(clone, 'clone.pt') # 4.3KB on disk