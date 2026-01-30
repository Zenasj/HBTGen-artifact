import torch
a = torch.randn(2).to('xpu')
a.data_ptr()
# one possible output is
-23453392437248
# this is inconsistent with storage.data_ptr()
a.untyped_storage().data_ptr()
# one possible output is
18446720620317114368

import torch
a = torch.randn(2).to('xpu')
a.data_ptr()
# one possible output is
18446720620317114368
# this is consistent with storage.data_ptr()
a.untyped_storage().data_ptr()
# one possible output is
18446720620317114368