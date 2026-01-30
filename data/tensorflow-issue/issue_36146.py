class CustomMapping(Mapping):

  def __init__(self, **kwargs):
    self.mapping = kwargs
  
  def __getitem__(self, key):
    return self.mapping[key]
  
  def __iter__(self):
    return iter(self.mapping)

  def __len__(self):
    return len(self.mapping)