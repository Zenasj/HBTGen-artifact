import torch.nn as nn

import torch
embedding = torch.nn.EmbeddingBag(num_embeddings=128, embedding_dim=32)

inputs = torch.randint(low=0, high=128, size=(1, 10))
device = "cuda" if torch.cuda.is_available() else "cpu"

model = embedding.to(device=device)
example_inputs = inputs.to(device)

# use c abi when 
with torch._inductor.config.patch({"aot_inductor.abi_compatible": True}):
    so_path = torch._export.aot_compile(
        model,
        (example_inputs, ),
        options={"aot_inductor.output_path": r"/tmp/test_model.so"})