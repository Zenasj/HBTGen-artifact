class DefaultCollate:
  def __call__(self, batch):
     pass

# Custom collates
class CustomCollate(DefaultCollate):
  def __call__(self, batch):
    if isinstance(batch[0], CustomType):
      pass
    else:
      return super().__call__(batch)