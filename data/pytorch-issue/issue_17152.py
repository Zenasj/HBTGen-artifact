import torch                                                                                                                           
a = torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [])                                                                               
a.clone()  #error
a.coalesce() #error