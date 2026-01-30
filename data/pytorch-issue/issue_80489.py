# repro.py
import torch
from functorch import vmap
x = torch.randn(2, 3, 5)
vmap(lambda x: x, out_dims=3)(x)

import torchtext
torchtext._torchtext._build_vocab_from_text_file_using_python_tokenizer("doesnotexist", 10, 10)