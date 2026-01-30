import torch 

unique_classes, counts = torch.unique(
    input=torch.tensor([0, 0, 2, 1, 1, 1]).long(), 
    sorted=False, 
    return_counts=True
)
print(unique_classes, counts)

(tensor([1, 2, 0]), tensor([3, 1, 2]))

tensor([0, 2, 1])

tensor([1, 2, 0])

x = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
x.unique_consecutive(return_counts=True)

(tensor([1, 2, 3, 1, 2, 3, 1, 2, 3]), tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]))

x.sort()[0].unique_consecutive(return_counts=True)

(tensor([1, 2, 3]), tensor([3, 3, 3]))