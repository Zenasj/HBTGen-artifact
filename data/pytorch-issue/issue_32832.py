import torch
    
foo = torch.randn(2).cuda()
foo = foo[range(10000)]  # should not be legal

bar = torch.randn(2) # goes fine
bar.cuda()  # crashes because of foo indexing