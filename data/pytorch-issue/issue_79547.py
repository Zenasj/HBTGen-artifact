from torchdata.datapipes.iter import IterableWrapper

source_dp = IterableWrapper(range(10))
it1 = iter(source_dp)
next(it1)
it2 = iter(source_dp)
next(it2)
next(it1)