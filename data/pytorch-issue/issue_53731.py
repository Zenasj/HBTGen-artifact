def __setitem__(self, key, storage_ref):
    with self.lock:
        dict.__setitem__(self, key, storage_ref)
        if len(self) > self.limit:
            self.free_dead_references()