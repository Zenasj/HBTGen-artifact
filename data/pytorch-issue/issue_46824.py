from torch import Tensor

class MyTensor(Tensor):
    pass

a = MyTensor([1,2,3])
a.new_ones(320)

from torch import Tensor

class MyTensor(Tensor):
    pass

a = MyTensor([1,2,3])
a.new_zeros(320).shape

(320,)