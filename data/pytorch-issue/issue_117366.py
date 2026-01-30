import torch

n_embd = 10
n_hidden = 100
vocab_size=27
block_size=3

# init weights
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g) * 0.1

parameters = [C, W1, b1]
for p in parameters:
  p.requires_grad = True

# training set
Xtr, Ytr = torch.randint(0, 27, (1000, 3)), torch.randint(0, 27, (1000,), generator=g)
batch_size = 32
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# forward pass
emb = C[Xb].flatten(1)
logits = emb @ W1 + b1
logit_maxes = logits.max(dim=1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(batch_size), Yb].mean()

for p in parameters:
  p.grad = None
# retaining all intermediary grads because I was planning on comparing them with the manual computation.
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits,
         emb]:
  t.retain_grad()
loss.backward()

dcounts = counts
for i, count in enumerate(dcounts):
    dcounts[i] = dcounts[i]/dcounts[i].sum()

dcounts /= 32 # Where runtime error occurs