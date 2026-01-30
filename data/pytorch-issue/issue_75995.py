source_dp = IterableWrapper(range(10))
cdp1, cdp2 = source_dp.fork(num_instances=2)
it1, it2 = iter(cdp1), iter(cdp2)
list(it1)  # [0, ..., 9]
list(it2)  # [0, ..., 9]
it1, it2 = iter(cdp1), iter(cdp2)
it3 = iter(cdp1)  # Basically share the same reference as `it1`, doesn't reset because `cdp1` hasn't been read since reset
next(it1)  # returns 0, no error
next(it2)  # returns 0, no error
next(it3)  # returns 1
# The next line resets all ChildDataPipe because `cdp2` has started reading
it4 = iter(cdp2)
next(it3)  # returns 0
list(it4)  # [0, ..., 9]

source_dp = IterableWrapper(range(10))
cdp1, cdp2 = source_dp.fork(num_instances=2)
it1, it2 = iter(cdp1), iter(cdp2)
list(it1)  # [0, ..., 9]
list(it2)  # [0, ..., 9]
it1, it2 = iter(cdp1), iter(cdp2)
it3 = iter(cdp1)  # This invalidates `it1` and `it2`
next(it1)  # raises an error
next(it2)  # raises an error
next(it3)  # returns 0
# The next line should not invalidate anything, as there was no new iterator created
# for `cdp2` after `it2` was invalidated
it4 = iter(cdp2)
next(it3)  # returns 1, no error here
list(it4)  # [0, ..., 9]