import torch
torch.set_printoptions(precision=20)
tensor1 = torch.load('tensor1.t')
tensor2 = torch.load('tensor2.t')
result = tensor1 @ tensor2
print(result[6, 0]) # incorrect
print(sum(tensor1[6] * tensor2[:, 0])) # correct
print(result[4, 0]) # correct

tensor(-0.00154018448665738106)
tensor(-0.00154018541797995567)
tensor(-0.00154018541797995567)