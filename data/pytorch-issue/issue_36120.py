import torch.nn as nn

from time import time
import torch

def main():
  batch_size = 2048
  num_features = 100000
  query_nnz = 100
  embed_size = 64

  ref_embedding = torch.nn.Embedding(num_features, embed_size, sparse=True).cuda()

  indices = torch.randint(0, high=num_features, size=(batch_size, query_nnz), device="cuda")
  grad = torch.rand(batch_size, query_nnz, embed_size, device="cuda")


  torch.cuda.synchronize()
  start = time()
  for _ in range(100):
    ref_embedding.weight.grad = None
    ref_lookup = ref_embedding(indices)
    ref_lookup.backward(grad)
  torch.cuda.synchronize()
  stop = time()
  print(F"Elapsed time {(stop - start) * 1000.:.1f} ms.")


if __name__ == '__main__':
  main()