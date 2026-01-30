import torch

with torch.no_grad():
   with torch.cpu.amp.autocast():
       model = torch.jit.trace(model, input_sample, check_trace=False)
# it can inference here
for i in range(10):
   model(sample)
with torch.no_grad():
   with torch.cpu.amp.autocast():
       model.save("jit_bf16.ckpt")
       model = torch.jit.load("jit_bf16.ckpt")
       for i in range(10):
            # it will reprot error when i = 1
           model(sample)