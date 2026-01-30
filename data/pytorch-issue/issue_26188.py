import gc
import numpy as np
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, *embeddings):
        super(MyModel, self).__init__()
        self.embeddings = nn.ModuleList()
        for embedding in embeddings:
            embedding = torch.as_tensor(embedding)
            embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
            self.embeddings.append(embedding)

device = "cuda"
embeddings = [np.zeros((5000000, 1000))] * 3
model = MyModel(*embeddings)

try:
    model = model.to(device)
except RuntimeError: # CUDA out of memory
    print("GPU memory overflow. Use CPU instead.")
    del model
    gc.collect()
    torch.cuda.empty_cache()