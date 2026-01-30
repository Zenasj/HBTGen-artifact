python
from sys import argv
import torch
print(torch.__version__)
from tqdm import tqdm

N = int(argv[1])

a, b = [torch.randn([N]) for _ in (0, 1)]
for _, i in tqdm(enumerate(range(1000000))):
  b.copy_(a)

python
from sys import argv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class Dummy(Dataset):

  def __len__(self):
    return 1000000

  def __getitem__(self, idx):
    return torch.zeros([3, 32, 32])

loader = DataLoader(Dummy(), num_workers=int(argv[1]), pin_memory=int(argv[2]), batch_size=int(argv[3]))
loader = iter(loader)
next(loader)
print(torch.__version__)
for _, im in tqdm(enumerate(loader)):
  pass