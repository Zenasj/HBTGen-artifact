import torch

@torchdynamo.optimize("inductor")
def user_code():
  module = module.cuda()
  print(module)
  torch.save(module)

def train(model,data):
  for batch in data:
    model(data)
...