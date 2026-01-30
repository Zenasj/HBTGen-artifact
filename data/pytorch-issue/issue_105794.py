import torch

idx1 = torch.tensor([[1], [2], [2]])
print(idx1)
idx2 = torch.tensor([[1, 2, 2]]).t()
print(idx2)

# idx1 and idx2 are the same but the output results are different. Why?

a = torch.tensor([[0.6280, 0.5672, 0.3760],
                  [0.4340, 0.9902, 0.1539],
                  [0.4585, 0.2638, 0.6983]])

print(a.gather(0, idx1))
print(a.gather(0, idx2))

# this is output
tensor([[1],
        [2],
        [2]])
tensor([[1],
        [2],
        [2]])
tensor([[0.5672],
        [0.3760],
        [0.3760]])
tensor([[0.4340],
        [0.4585],
        [0.4585]])