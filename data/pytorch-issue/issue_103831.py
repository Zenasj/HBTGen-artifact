import torch
import torch.nn as nn

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

class Containing(nn.Module):
  def __init__(self, buflist):
    super().__init__()
    self.buflist = buflist
  def forward(self):
    for x in self.buflist:
      print(x.shape)

print(torch.__version__) #2.1.0.dev20230617+cu118
x = Containing(BufferList((torch.rand((5,5)), torch.rand((5,5)))))
x() # works
torch.compile(x, backend="eager")() # crashes