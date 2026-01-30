if isinstance(f, mmap.mmap):
    f = memoryview(f)