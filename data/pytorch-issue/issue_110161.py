import torch

dataset = [torch.nested.nested_tensor([torch.randn(5)]) for _ in range(100)]

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    collate_fn=lambda x: x,
)

for data in loader:
    print(data)