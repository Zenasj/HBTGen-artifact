import torch

from torch.utils.data.datapipes.iter import IterableWrapper, TarArchiveReader, FileLoader, Shuffler

# instantiate dp from any tar archive, 
# e.g. https://github.com/pytorch/data/blob/main/examples/vision/fakedata/caltech101/101_ObjectCategories.tar.gz
dp = FileLoader(dp)
dp = TarArchiveReader(dp)
dp = Shuffler(dp)

_, buffer = next(iter(dp))
buffer.read()