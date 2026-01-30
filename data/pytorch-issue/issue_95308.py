py
import torch

torch.manual_seed(420)

device = 'cuda'

t = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 3, 0]]).to(device)
index = torch.tensor([[2, 1, 0], [0, 2, 2]]).to(device)
src = torch.tensor([[0, 1, 2], [3, 4, 5]]).to(device)
t.index_put_((index,), src, accumulate=True)
# RuntimeError: linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel() 
# INTERNAL ASSERT FAILED at 
# "/opt/conda/conda-bld/pytorch_1672906354936/work/aten/src/ATen/native/cuda/Indexing.cu":389, 
# please report a bug to PyTorch. number of flattened indices did not match number of elements in the value tensor: 18 vs 6