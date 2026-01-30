import torch

torch.manual_seed(1)
weights = torch.load('weights.pt')
N, S = weights.shape[0], 4096
num_trials = 100
for trial in range(1, num_trials + 1):
  print('Starting trial %d / %d' % (trial, num_trials))
  weights[weights < 0] = 0.0
  samples = weights.multinomial(S, replacement=True)
  sampled_weights = weights[samples]
  assert sampled_weights.min() > 0