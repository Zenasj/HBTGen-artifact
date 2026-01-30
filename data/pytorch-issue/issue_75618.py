from torchdata.datapipes.iter import IterableWrapper, Mapper

def no_op(x):
    return x

def exception_when_one(x):
    if x == 1:
        raise RuntimeError("x cannot equal to 1.")
    return x

map_dp = IterableWrapper(range(10))
map_dp = Mapper(map_dp, no_op)
map_dp = Mapper(map_dp, exception_when_one)
map_dp = Mapper(map_dp, no_op)
print(list(map_dp))