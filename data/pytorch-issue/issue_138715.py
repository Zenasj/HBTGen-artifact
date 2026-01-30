import torch

# broken
self.encoder.compile()

# works
self.encoder = torch.compile(self.encoder)

# broken (same error in previous repro)
# self.encoder.compile(backend='inductor')

# works 
# self.encoder = torch.compile(self.encoder)

# broken (same error as previous repro)
for name, child in self.encoder.named_children():
    if isinstance(child, Float8Linear): 
        new_child = torch.compile(child)
        setattr(self.encoder, name, new_child)

m.encoder.compile()  # compile the encoder, which is just a Sequential()
m = FSDP(m) # lightning training later wraps the module with FSDP