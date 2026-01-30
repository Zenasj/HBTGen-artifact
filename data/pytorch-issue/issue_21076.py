import torch

In [27]: x = torch.randn(3)

In [28]: x.is_pinned()
Out[28]: False

In [29]: x = x.pin_memory()

In [30]: x.is_pinned()
Out[30]: True

In [31]: x.pin_memory().storage().data_ptr()
Out[31]: 139755750687744

In [32]: x.storage().data_ptr()
Out[32]: 139755182358528