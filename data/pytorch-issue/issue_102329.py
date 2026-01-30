#manual repro, as repro tools don't work
import torch
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.config.verbose=True
print(torch.__version__)

def fw_bw(*args):
  a,b,c = args
  print(len(args))
  pass

class Trainer():
  def __init__(self, fw_b):
    self._fw_bw = fw_b

trainer = Trainer(fw_bw)

def outer(trainer):
  trainer._fw_bw(1,2,3)

print("Normal")
outer(trainer)
print("Compiled")
torch.compile(outer, backend="eager")(trainer)

fw_b

trainer._fw_bw

self