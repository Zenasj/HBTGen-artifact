import scipy.io

from torchdata.datapipes.iter import FileLoader, IterableWrapper, Mapper

file = "foo.mat"
scipy.io.savemat(file, dict(foo="bar"))

dp = IterableWrapper((file,))
dp = FileLoader(dp)
dp = Mapper(dp, scipy.io.loadmat, input_col=1)

next(iter(dp))

import scipy.io

from torchdata.datapipes.iter import FileLoader, IterableWrapper, Mapper

file = "foo.mat"
scipy.io.savemat(file, dict(foo="bar"))


def loadmat(stream_wrapper):
    return scipy.io.loadmat(stream_wrapper.file_obj)


dp = IterableWrapper((file,))
dp = FileLoader(dp)
dp = Mapper(dp, loadmat, input_col=1)

data = next(iter(dp))
assert data[1]["foo"] == "bar"