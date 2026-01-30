class CachingIterator:  # depends on CachedIterable so cannot be nested class.
    def __init__(self, ds: CachedIterable):
        self.ds = ds
        self.pos = -1
        self.raise_num = 0

    def __next__(self) -> Any:
        if self.raise_num == 1:
            self.raise_num == 2
            raise StopIteration

        if self.ds.itrt is not None:
            i = next(self.ds.itrt, None)
            if i is not None:
                self.ds.cache.append(i)
                return i
            else:
                print("CachingIterator stopes")
                self.ds.itrt = None  # notify CachedIterable.__iter__ to yield cache.
                self.raise_num += 1
                raise StopIteration
        else:
            if self.pos + 1 < len(self.ds.cache):
                self.pos = self.pos + 1
                return self.ds.cache[self.pos]
            else:
                raise StopIteration