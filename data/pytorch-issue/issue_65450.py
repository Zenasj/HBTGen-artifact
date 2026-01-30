class ContainsRRef:
    def __init__(self, rref):
        self.rref = rref

    def foo(self):
        pass

    def __getstate__(self):
      # explicitly skipping pickling RRef
      return {}