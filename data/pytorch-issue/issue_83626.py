import torch

# This should not be re-inplaced at all; the op broadcasts, so this would require resizing the self tensor
torch.add(tensor[1, 4], tensor[4, 4])

# This should not be re-inplaced, because the inplace and out-of-place variants of the op return different dtypes
torch.ge(a, b)
# However, this means that today when functionalization functionalists a `torch.ge_(a, b)` call, reinplacing won't properly de-functionalize it. I mentioned that optimization is worth adding later in the comments