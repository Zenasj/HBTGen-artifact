print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import torch
cuda = torch.cuda.is_available()
mps = torch.backends.mps.is_available()
if cuda:
    device = torch.device("cuda")
elif mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]])))