source_dp = IterableWrapper(range(10))
it1 = iter(source_dp)
list(it1)  # [0, 1, ..., 9]
it1 = iter(source_dp)
next(it1)  # 0
it2 = iter(source_dp)  
next(it2)  # returns 0
next(it1)  # returns 1

source_dp = IterableWrapper(range(10))
it1 = iter(source_dp)
list(it1)  # [0, 1, ..., 9]
it1 = iter(source_dp)  # This doesn't raise any warning or error
next(it1)  # 0, works because it is a new iterator
it2 = iter(source_dp)
next(it2) # returns 0, invalidates `it1`
next(it1)  # This raises an error

source_dp = IterableWrapper(range(10))
zip_dp = source_dp.zip(source_dp)
list(zip_dp)  # [(0, 0), ..., (9, 9)]

source_dp = IterableWrapper(range(10))
zip_dp = source_dp.zip(source_dp)
list(zip_dp)  # This raises an error because there are multiple references to `source_dp`

source_dp = IterableWrapper(range(10))
dp1, dp2 = source_dp.fork(2)
zip_dp = dp1.zip(dp2)
list(zip_dp)  # [(0, 0), ..., (9, 9)]

dp = IterableWrapper(range(10))
it1 = iter(dp)
it2 = iter(dp)  # Only this one will work

# Assuming some `dp` was previously defined in another cell
it1 = iter(dp)  # Unless it just returns `self`, this will raise an error in re-runs (if we disallow creation of multiple iterators)
# And if it does return `self`, then the users need to call something else to reset the DataPipe to read from the beginning instead.